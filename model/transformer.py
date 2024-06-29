import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_embeddings = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, emb_size)
        x += self.position_embeddings  # 加上位置嵌入
        return x

# 定义Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        assert self.head_dim * num_heads == emb_size, "Embedding size must be divisible by number of heads"

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.emb_size)
        out = self.fc_out(out)
        return out

# 定义Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1, forward_expansion=4):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(attn_out + x)
        ff_out = self.feed_forward(x)
        x = self.norm2(ff_out + x)
        return x

# 定义Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size, num_classes, depth, num_heads, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(emb_size, num_heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_size)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        out = self.classifier(cls_token)
        return out
    
    
if __name__ == "__main__":
    vit = VisionTransformer(3, 16, 768, 224, 100, 6, 8)
    test_data = torch.rand(128, 3, 224, 224)
    print(vit(test_data))