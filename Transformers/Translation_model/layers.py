import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # GELU tends to perform better but is more computationally expensive than ReLU
        self.gelu = nn.GELU()
        # self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Linear(d_model, d_ff, bias=True) 
        self.W2 = nn.Linear(d_ff, d_model, bias=True) 

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.W1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.W2(x)
        x = self.dropout(x)
        
        return x + residual

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout(x) 
        return x + residual

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, mask=None):
        residual = x
        x = self.norm(x)
        x = self.attn(x, encoder_output, encoder_output, mask)
        return residual + self.dropout(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionBlock(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.self_attn(x, tgt_mask)
        x = self.cross_attn(x, encoder_output, src_mask)
        x = self.ffn(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.ffn(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]