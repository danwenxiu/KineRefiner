from torch import nn


# class Sum_Attention(nn.Module):
#
#
#     def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  mode='spatial'):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim_in // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim_in, dim_out)
#         self.mode = mode
#         self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x,att_map,weight):     ##x=（4,243,17,128）
#         B, T, J, C = x.shape
#
#         qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
#                                                                                            5)  # (3, B, H, T, J, C)
#
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         B, H, T, J, C = q.shape
#         qt = q.transpose(2, 3)  # (B, H, J, T, C)
#         kt = k.transpose(2, 3)  # (B, H, J, T, C)
#         vt = v.transpose(2, 3)  # (B, H, J, T, C)
#
#         attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#
#         attn = weight*attn + (1-weight)*att_map
#
#         attn = self.attn_drop(attn)
#         x = attn @ vt  # (B, H, J, T, C)
#         x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)    ##x=（4,243,17,128）
#         return x


import torch
class Sum_Attention(nn.Module):   #1月21有点效果的   40.4  33.8    下面的哪个是基于这个的进一步曾强
    def __init__(self, dim, out_dim, num_heads,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., mode='temporal'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # ===== 论文中的 µ：per-head 可学习 =====
        self.mu = nn.Parameter(torch.zeros(num_heads))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ===== 诊断用（不影响 forward 接口）=====
        self.last_attn = None

    def forward(self, x, att_map, weight):
        """
        x:        (B, T, J, C)
        att_map:  (B, H, J, T, T)  = M_F→P · M_P→F
        weight:   scalar（保留接口，不再作为论文核心变量）
        """
        B, T, J, C = x.shape
        H = self.num_heads

        # ======================================================
        # QKV（与你给的代码一字不差）
        # ======================================================
        qkv = self.qkv(x).reshape(
            B, T, J, 3, H, C // H
        ).permute(3, 0, 4, 1, 2, 5)  # (3, B, H, T, J, C_h)

        q, k, v = qkv[0], qkv[1], qkv[2]
        B, H, T, J, C_h = q.shape

        qt = q.transpose(2, 3)  # (B, H, J, T, C_h)
        kt = k.transpose(2, 3)
        vt = v.transpose(2, 3)

        # ======================================================
        # 1️⃣ M_F→F：标准 self-attention（你原来的）
        # ======================================================
        attn_ff = (qt @ kt.transpose(-2, -1)) * self.scale  # (B,H,J,T,T)
        attn_ff = attn_ff.softmax(dim=-1)
        attn_ff = self.attn_drop(attn_ff)

        # ======================================================
        # 2️⃣ M_agg：显式使用 F→P · P→F（来自 MIBlock）
        # ======================================================
        attn_agg = att_map  # (B,H,J,T,T)

        # ======================================================
        # 3️⃣ 论文公式融合
        # M = M_agg + μ · M_F→F
        # ======================================================
        mu = torch.sigmoid(self.mu).view(1, H, 1, 1, 1)
        attn = attn_agg + mu * attn_ff
        attn = self.attn_drop(attn)

        # ======================================================
        # 4️⃣ Value 聚合（与你原代码一致）
        # ======================================================
        x = attn @ vt  # (B,H,J,T,C_h)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C_h * H)

        x = self.proj(x)
        x = self.proj_drop(x)

        # ======================================================
        # 5️⃣ 缓存融合注意力（用于诊断 / 可视化）
        # ======================================================
        self.last_attn = {
            "attn_ff": attn_ff.detach(),
            "attn_agg": attn_agg.detach(),
            "attn_fused": attn.detach(),
            "mu": mu.detach()
        }

        return x











