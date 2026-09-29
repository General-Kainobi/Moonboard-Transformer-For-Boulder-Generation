import torch
import torch.nn as nn
import math
from config import VOCAB_SIZE, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT, MAX_SEQ_LEN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class MoonBoardTransformer(nn.Module):
    """
    Causal Transformer (Decoder-Only) for MoonBoard sequence generation.
    Combines discrete token embeddings with continuous (dx, dy) spatial embeddings.
    """ 
#dx,dy differenze between the x,y of a single move
    def __init__(self):
        super().__init__()
        
        self.d_model = EMBEDDING_DIM
        
        # Token embedding (for discrete holds)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        
        # Continuous embedding for (dx, dy)
        self.spatial_projection = nn.Linear(2, self.d_model)
        
        # We combine them, so maybe project sum or concat?
        # A simple sum of token embedding + spatial embedding + positional encoding is standard.
        self.pos_encoder = PositionalEncoding(self.d_model, MAX_SEQ_LEN)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=NUM_HEADS,
            dim_feedforward=self.d_model * 4,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=NUM_LAYERS)
        
        # Output head to predict next token
        self.fc_out = nn.Linear(self.d_model, VOCAB_SIZE)
        
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask to prevent looking ahead in the sequence."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tokens: torch.Tensor, dx_dy: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        tokens: [batch_size, seq_len]
        dx_dy: [batch_size, seq_len, 2]
        src_key_padding_mask: [batch_size, seq_len] (Boolean tensor where True means ignore)
        """
        batch_size, seq_len = tokens.size()
        
        # Embeddings
        tok_emb = self.token_embedding(tokens) # [batch_size, seq_len, d_model]
        spat_emb = self.spatial_projection(dx_dy) # [batch_size, seq_len, d_model]
        
        # Combine
        x = tok_emb + spat_emb
        
        # Positional encoding (expects [seq_len, batch_size, d_model] if batch_first=False)
        # We are using batch_first=True for TransformerEncoder, but our PositionalEncoding expects batch dim 1.
        x = x.transpose(0, 1) # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # [batch_size, seq_len, d_model]
        
        causal_mask = self.generate_square_subsequent_mask(seq_len, tokens.device)
        
        # If we pass src_key_padding_mask to PyTorch Transformer, False = keep, True = mask out
        # In train_supervised, it is already True for PAD_TOKEN, so we pass it directly.
        padding_mask = src_key_padding_mask
        
        # Transformer forward
        out = self.transformer(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask,
            is_causal=True
        ) # [batch_size, seq_len, d_model]
        
        logits = self.fc_out(out) # [batch_size, seq_len, vocab_size]
        
        return logits
