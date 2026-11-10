"""randsvd: the one input family of the QR audit that is not the organizers' own.

A = U diag(s) V^T with U, V random orthogonal and s = logspace(0, -log10(kappa), n),
so kappa_2(A) = kappa before the cast to float32. This is the classical
"randsvd" test matrix (Higham, Accuracy and Stability of Numerical Algorithms,
2nd ed., Section 28.3, and MATLAB's gallery('randsvd')). Every other family in
qr/sweep.csv is produced by the organizers' generate_input in reference-kernels.
"""
import math
import torch


def _rand_orthogonal(batch, n, gen, device):
    a = torch.randn((batch, n, n), device=device, dtype=torch.float64, generator=gen)
    q, _ = torch.linalg.qr(a)
    return q


def randsvd(batch, n, kappa, seed, device="cuda", dtype=torch.float32):
    """Batch of n x n matrices with exact target 2-norm condition number `kappa`.

    Built in float64, then cast to the competition dtype (float32). The cast
    perturbs kappa slightly; qr/kappa_bisect.csv records the realized value
    (`realized_kappa`) at each threshold.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    u = _rand_orthogonal(batch, n, gen, device)
    v = _rand_orthogonal(batch, n, gen, device)
    if n == 1:
        s = torch.ones((1,), device=device, dtype=torch.float64)
    else:
        s = torch.logspace(0.0, -math.log10(kappa), n, device=device, dtype=torch.float64)
    a = (u * s.unsqueeze(-2)) @ v.transpose(-1, -2)
    return a.to(dtype).contiguous()


