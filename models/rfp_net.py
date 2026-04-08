 

import torch
import torch.nn as nn
import torch.nn.functional as F

class DFPM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, D)
        weights = self.gate(x)
        return x * weights  # highlight important features

class TCPB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, dilation=2, padding=2)

        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # (B, T, D) → (B, D, T)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        return x.permute(0, 2, 1)

class FITA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.align = nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, x, prev_feedback=None):
        if prev_feedback is None:
            prev_feedback = x

        aligned, _ = self.align(x, prev_feedback, prev_feedback)
        return x + aligned

class RFPEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dfpm = DFPM(dim)
        self.tcpb = TCPB(dim)
        self.fita = FITA(dim)
    def forward(self, x):
        x = self.dfpm(x)
        x = self.tcpb(x)
        x = self.fita(x)
        return x


class DMSM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)

        # ✅ FIXED GATE
        self.gate = nn.Sequential(
            nn.Linear(dim * 6, dim * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # → (B, D, T)

        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)

        fused = torch.cat([f1, f3, f5], dim=1)  # (B, 3D, T)

        avg_pool = torch.mean(fused, dim=2)
        max_pool, _ = torch.max(fused, dim=2)

        context = torch.cat([avg_pool, max_pool], dim=1)  # (B, 6D)

        gate = self.gate(context).unsqueeze(-1)  # (B, 3D, 1)

        out = fused * gate  # ✅ MATCHES NOW

        return out.permute(0, 2, 1)  # → (B, T, 3D)


class TGR(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.conv = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)

        # graph-like propagation
        x = x + attn_out

        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x

class CATKDecoder(nn.Module):
    def __init__(self, dim, num_classes=3, k=3):
        super().__init__()

        self.fc = nn.Linear(dim, num_classes)
        self.k = k

    def forward(self, x):
        logits = self.fc(x)
        return logits

class RFPDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dmsm = DMSM(dim)
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        self.tgr = TGR(dim)
        self.catk = CATKDecoder(dim)

    def forward(self, x):
        x = self.dmsm(x)  # (B, T, 384)
        x = self.proj(x)  # (B, T, 128) ✅ BACK TO NORMAL
        x = self.norm(x)
        x = self.tgr(x)

        return self.catk(x)

class RFPNet(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.encoder = RFPEncoder(dim)
        self.decoder = RFPDecoder(dim)

    def forward(self, fused_features):
        # ✅ FIX: Ensure 3D tensor
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)

        encoded = self.encoder(fused_features)
        output = self.decoder(encoded)

        return output
