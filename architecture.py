import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 3. TRM Block: Custom assembly because standard PyTorch uses LayerNorm
class TRMBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_model * 4)

    def forward(self, x):
        # Pre-norm architecture [cite: 40]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# 4. Updated TinyRecursiveARC (V3 - Paper Accurate)
class TinyRecursiveARC(nn.Module):
    def __init__(self, d_model=256, nhead=8, T=3, n_inner=6):
        super().__init__()
        self.T = T              # Outer recursion [cite: 180]
        self.n_inner = n_inner  # Inner latent update [cite: 180]
        
        self.token_emb = nn.Embedding(11, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 900, d_model))
        
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
        x = self.token_emb(grids.view(B, -1)) + self.pos_emb
        y = self.y_init.expand(B, -1, -1)
        z = self.z_init.expand(B, -1, -1)
        
        # Recursion steps
        for _ in range(self.T):
            y, z = self.latent_recursion(x, y, z)
            
        logits = self.color_head(y)
        pred_size = torch.sigmoid(self.size_head(z.mean(dim=1))) * 30.0
        return logits, pred_size

    def hard_predict(self, grids):
        self.eval()
        with torch.no_grad():
            logits, pred_size = self.forward(grids)
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
