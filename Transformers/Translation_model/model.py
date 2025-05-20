import torch
import torch.nn as nn
from layers import EncoderLayer, DecoderLayer, PositionalEncoding
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
            
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        causal_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & causal_mask.to(device)
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_emb = self.encoder_embed(src) * math.sqrt(self.d_model) 
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        tgt_emb = self.decoder_embed(tgt) * math.sqrt(self.d_model) 
        tgt_emb =self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            
        return self.fc_out(dec_output)
            