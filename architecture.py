import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. RMSNorm: Skips mean subtraction for recursive stability 
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# 2. SwiGLU: The paper's secret sauce for 7M models [cite: 40, 254]
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# 3. Rotary Positional Embeddings (RoPE)
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len=900, base=10000):
        super().__init__()
        # d_model is the hidden dimension of the model
        # We apply RoPE to each head, so we need head_dim = d_model / nhead
        # But RoPE is often applied to a fraction or full head_dim.
        # Assuming full head_dim application.
        # However, the rotation matrix depends on the position index.
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        # Cache cos and sin
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, nhead, seq_len, head_dim]
        # returns cos, sin with shape [1, 1, seq_len, head_dim] (broadcastable)
        if seq_len > self.max_seq_len:
             # Recompute if seq_len exceeds cache
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
            return cos, sin
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, nhead, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 4. Custom Self-Attention with RoPE
class TRMAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=900)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2) # [B, nhead, L, head_dim]
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False) # Not causal for bidirectional grid

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)

# 5. TRM Block: Custom assembly because standard PyTorch uses LayerNorm
class TRMBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        # Replaced nn.MultiheadAttention with TRMAttention (RoPE support)
        self.attn = TRMAttention(d_model, nhead)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_model * 4)

    def forward(self, x):
        # Pre-norm architecture [cite: 40]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 6. Updated TinyRecursiveARC (V3 - Paper Accurate with RoPE)
class TinyRecursiveARC(nn.Module):
    def __init__(self, d_model=256, nhead=8, T=3, n_inner=6):
        super().__init__()
        self.T = T              # Outer recursion [cite: 180]
        self.n_inner = n_inner  # Inner latent update [cite: 180]
        
        self.token_emb = nn.Embedding(11, d_model)
        # Removed absolute positional embeddings (self.pos_emb) as per RoPE implementation
        
        # Paper says 2 layers is optimal to prevent overfitting [cite: 156-158]
        self.layers = nn.ModuleList([TRMBlock(d_model, nhead) for _ in range(2)])
        
        # Heads without bias as per paper [cite: 40]
        self.color_head = nn.Linear(d_model, 10, bias=False)
        self.size_head = nn.Linear(d_model, 2, bias=False)
        
        self.y_init = nn.Parameter(torch.zeros(1, 900, d_model))
        self.z_init = nn.Parameter(torch.zeros(1, 900, d_model))

    def transformer_forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def latent_recursion(self, x, y, z):
        for _ in range(self.n_inner):
            z = self.transformer_forward(x + y + z) # Update z [cite: 99]
        y = y + self.transformer_forward(y + z) # Update y [cite: 99]
        return y, z

    def forward(self, grids):
        B = grids.shape[0]
        # Flatten grid: [B, H, W] -> [B, H*W]
        flat_grids = grids.view(B, -1)
        L = flat_grids.shape[1]

        # Only token embedding, no pos_emb added here (handled by RoPE in attention)
        x = self.token_emb(flat_grids)

        # If L != 900, we need to handle y_init and z_init
        if L != 900:
             # Option 1: Interpolate/Resample (complex)
             # Option 2: Slice/Pad (simple)
             # The model is trained on 30x30=900. If input is smaller, we should probably pad it to 30x30 BEFORE calling forward.
             # However, dataset collate_fn pads to 30x30.
             # Why is L != 900?
             # Ah, hard_predict might be called with raw unpadded grids in eval script?
             pass

        y = self.y_init.expand(B, -1, -1)
        z = self.z_init.expand(B, -1, -1)
        
        # Match sequence length if needed (e.g. for variable size inference, though ARC is usually padded to 30x30)
        if x.shape[1] != y.shape[1]:
            # If x is smaller (e.g. 2x2 grid = 4 tokens), but y_init is 900.
            # We must run on 900 tokens if we want to use the learned y_init/z_init directly?
            # Or we should slice y_init?
            # The paper TRM uses fixed size or padding.
            # Let's assume we must pad input to 30x30 if it isn't already.
            # But wait, RoPE handles variable length.
            # If we allow variable length, y_init must be sliced.
            if x.shape[1] < y.shape[1]:
                 y = y[:, :x.shape[1], :]
                 z = z[:, :x.shape[1], :]

        predictions = []
        # Recursion steps
        for _ in range(self.T):
            y, z = self.latent_recursion(x, y, z)
            
            # Deep Supervision: Predict at each step
            logits = self.color_head(y)
            pred_size = torch.sigmoid(self.size_head(z.mean(dim=1))) * 30.0
            predictions.append((logits, pred_size))

        return predictions

    def hard_predict(self, grids):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(grids)
            # Take the final prediction
            logits, pred_size = predictions[-1]

            color_preds = torch.argmax(logits, dim=-1).view(-1, 30, 30)
            preds = []
            for i in range(grids.shape[0]):
                h_raw, w_raw = pred_size[i, 0], pred_size[i, 1]
                # Catching NaNs before they crash the laptop
                if torch.isnan(h_raw) or torch.isnan(w_raw):
                    h, w = 1, 1
                else:
                    h = int(torch.round(h_raw).clamp(1, 30).item())
                    w = int(torch.round(w_raw).clamp(1, 30).item())
                preds.append(color_preds[i, :h, :w])
            return preds
