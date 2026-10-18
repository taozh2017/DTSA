import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if (num_heads != 1 or dim_head != dim) else nn.Identity()

    def forward(self, queries, keys):
        q = self.to_q(queries)

        k, v = self.to_kv(keys).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale * self.temperature
        attn = attn.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerDecoderBlock(nn.Module):
    """公共基础模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )


class Img_TransformerDecoderLayer(TransformerDecoderBlock):
    def __init__(self, input_size, in_channels, d_model, nhead=8, dropout=0.1, patch_size=7):
        super().__init__(d_model, nhead, dropout)

        patch_dim = in_channels * patch_size ** 2
        self.patch_size = patch_size
        self.text_repeat = (input_size // patch_size) ** 2
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, d_model)
        )
        self.self_attn = Attention(d_model, nhead, attn_drop=dropout, proj_drop=dropout)
        self.cross_attn = CA_Attention(d_model, nhead, dropout=dropout)

        self.to_out = nn.Sequential(
            nn.Linear(d_model, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=input_size//patch_size, p1=patch_size, p2=patch_size)
        )

        self.text_proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "text_storage_buffer",
            torch.empty(0, d_model, dtype=torch.float32)
        )

        self._temp_fingerprints = set()

    def _get_tensor_fingerprint(self, tensor: torch.Tensor) -> tuple:
        """快速计算特征指纹，用于去重"""
        tensor_norm = F.normalize(tensor, p=2, dim=-1)
        return (round(tensor_norm.mean().item(), 6), round(tensor_norm.sum().item(), 6))

    def add_image_text(self, tensor_batch: torch.Tensor):
        """每 step 自动新增特征，动态扩容 buffer，有新增才打印"""
        if not self.training:
            return

        new_feats = []
        for tensor in tensor_batch:
            text_feat = tensor.squeeze(dim=0).detach()
            fingerprint = self._get_tensor_fingerprint(text_feat)

            if fingerprint not in self._temp_fingerprints:
                self._temp_fingerprints.add(fingerprint)
                new_feats.append(text_feat)

        if len(new_feats) > 0:
            new_feats_tensor = torch.stack(new_feats).to(self.text_storage_buffer.device)
            self.text_storage_buffer = torch.cat([self.text_storage_buffer, new_feats_tensor], dim=0)
            total = self.text_storage_buffer.shape[0]
            print(f"新增 {len(new_feats)} 个唯一文本特征，累计存储 {total} 个")

    def match_similar_text(self, unlab_image_patches, aggregation="mean"):
        """直接读取动态扩容后的 buffer（无零向量）"""
        if self.text_storage_buffer.shape[0] == 0:
            raise ValueError("文本特征存储为空！请先训练标注数据（text_feature不为None）")

        text_feats = self.text_storage_buffer

        if aggregation == "mean":
            unlab_global_feats = unlab_image_patches.mean(dim=1)
        elif aggregation == "cls":
            unlab_global_feats = unlab_image_patches[:, 0, :]
        else:
            raise ValueError("聚合方式仅支持 'mean' 或 'cls'")

        unlab_global_feats = F.normalize(unlab_global_feats, p=2, dim=1)
        text_feats_norm = F.normalize(text_feats, p=2, dim=1)
        similarity_matrix = torch.matmul(unlab_global_feats, text_feats_norm.T)

        max_similarities, matched_indices = torch.max(similarity_matrix, dim=1)
        matched_text_feats = text_feats[matched_indices]
        matched_text_tensors = matched_text_feats.unsqueeze(dim=1)

        return matched_text_tensors, max_similarities

    def forward(self, x, text_feature):
        q = self.to_patch(x)

        if text_feature is not None:
            self.add_image_text(text_feature)

            b = x.shape[0]
            matched_text, max_similarities = self.match_similar_text(q[b//2:])
            text_feature = torch.concat([text_feature, matched_text], dim=0)
        else:
            matched_text, max_similarities = self.match_similar_text(q)
            text_feature = matched_text

        B, N, _ = q.shape

        identity = q
        q = self.norm1(q)
        q = self.self_attn(q) + identity

        identity = q
        q = self.norm2(q)

        text_feature = self.text_proj(text_feature)
        text_feature = text_feature.repeat(1, self.text_repeat, 1)
        x = self.cross_attn(text_feature, q) + identity

        identity = x
        x = self.norm3(x)
        x = self.mlp(x) + identity

        return self.to_out(x)

class Cls_TransformerDecoderLayer(TransformerDecoderBlock):
    def __init__(self, input_size, in_channels, d_model, n_class, nhead=8, dropout=0.1, patch_size=14, c_width=4):
        super().__init__(d_model, nhead, dropout)

        patch_dim = c_width * patch_size ** 2
        self.n_class = n_class
        self.c_width = c_width
        self.text_repeat = in_channels // c_width
        self.to_patch = nn.Sequential(
            Rearrange('b (c cw) (h p1) (w p2) -> b (h w c) (p1 p2 cw)',
                      cw=c_width, p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, d_model)
        )

        self.self_attn = Attention(d_model, nhead, attn_drop=dropout, proj_drop=dropout)
        self.cross_attn = CA_Attention(d_model, nhead, dropout=dropout)

        self.to_out = nn.Sequential(
            nn.Linear(d_model, patch_dim),
            Rearrange('b k (h w c) (p1 p2 cw) -> b k (c cw) (h p1) (w p2)',
                     h=input_size//patch_size, c=in_channels//c_width,
                     p1=patch_size, p2=patch_size, cw=c_width)
        )

        self.text_proj = nn.Linear(d_model, d_model)

    def forward(self, x, text_features):
        B, C, H, W = x.shape
        q = self.to_patch(x)

        identity = q
        q = self.norm1(q)
        q = self.self_attn(q) + identity

        text_features = self.text_proj(text_features)
        text_features = repeat(text_features, 'b n d -> b n k d', k=self.text_repeat)
        text = rearrange(text_features, 'b c k d -> (b c) k d')

        q = repeat(q, 'b n d -> b c n d', c=self.n_class)
        q = rearrange(q, 'b c n d -> (b c) n d')

        identity = q
        q = self.norm2(q)
        x = self.cross_attn(text, q) + identity
        x = rearrange(x, '(b c) n d -> b c n d', b=B, c=self.n_class)

        identity = x
        x = self.norm3(x)
        x = self.mlp(x) + identity

        x = self.to_out(x)
        cls_attn = x.reshape(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])

        return cls_attn


