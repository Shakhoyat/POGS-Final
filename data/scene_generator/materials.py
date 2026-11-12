"""
Procedural materials for the analytic ray-caster.

Every texture is evaluated in the hit primitive's OWN frame (``R^T (x - t)``),
so it moves rigidly with the object. (The first generator evaluated textures in
world coordinates: a moving object slid underneath a fixed pattern, which
corrupts exactly the appearance features POGS tracks.)

Noise uses an integer lattice hash, so CPU and CUDA renders are identical.

Patterns (``Primitive.pattern``; ``texture_freq`` = features per metre,
``texture_amp`` = strength):
    sine     legacy sin(x) cos(y) modulation
    noise    fine fBm grain (plastic, glaze, skin)
    wood     grain streaks along the local x axis, colour -> color2
    split    local z > 0: color, below: color2 (two-tone can label)
    stripes  bands of color2 across local z
    blotch   fBm-thresholded patches of color2 (printed box art, apple skin)
    tiles    square tiles with color2 grout and per-tile shade (floor)
    brushed  noise stretched along local x (brushed metal)
"""

from __future__ import annotations

from typing import Tuple

import torch

PATTERNS = {"sine": 0, "noise": 1, "wood": 2, "split": 3, "stripes": 4, "blotch": 5, "tiles": 6, "brushed": 7}

_M = 0x7FFFFFFF


def _hash(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
    """Integer lattice hash -> [0, 1). All products stay below 2^62."""
    h = ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) & _M
    h = ((h >> 13) ^ h) & _M
    h = (h * 1274126177) & _M
    h = ((h >> 16) ^ h) & _M
    return h.float() / float(_M)


def value_noise(p: torch.Tensor) -> torch.Tensor:
    """Smooth value noise in [0, 1] at points ``p`` (N, 3) in lattice units."""
    i = torch.floor(p)
    f = p - i
    u = f * f * (3.0 - 2.0 * f)
    i = i.long()
    out = torch.zeros(p.shape[0], device=p.device)
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1.0 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1.0 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1.0 - u[:, 2]
                out = out + wx * wy * wz * _hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out


def fbm(p: torch.Tensor, octaves: int = 3) -> torch.Tensor:
    """Fractal sum of ``value_noise``, normalised to [0, 1]."""
    out, amp, norm = torch.zeros(p.shape[0], device=p.device), 1.0, 0.0
    for o in range(octaves):
        out = out + amp * value_noise(p * (2.0 ** o) + 17.0 * o)
        norm += amp
        amp *= 0.5
    return out / norm


def surface_color(scene, prim: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Albedo (N, 3) at world points ``pts`` lying on primitives ``prim``."""
    R, t, s = scene.R[prim], scene.t[prim], scene.scale[prim]
    loc = torch.einsum("nji,nj->ni", R, pts - t)            # metres, primitive frame
    loc_n = loc / s                                          # [-1, 1] across the primitive
    freq, amp = scene.tex_args[prim, 0], scene.tex_args[prim, 1]
    phase = scene.tex_args[prim, 2:5]
    pat = scene.pattern[prim]
    c1, c2 = scene.color[prim], scene.color2[prim]
    q = loc * freq[:, None] + phase

    n = fbm(q)                                               # shared grain
    col = c1.clone()
    mod = torch.ones_like(freq)

    def sel(name):
        return pat == PATTERNS[name]

    m = sel("sine")
    if m.any():
        mod[m] = 1.0 + amp[m] * (0.5 * torch.sin(q[m, 0]) * torch.cos(q[m, 1]))
    m = sel("noise")
    if m.any():
        mod[m] = 1.0 + 2.0 * amp[m] * (n[m] - 0.5)
    m = sel("wood")
    if m.any():
        w = fbm(q[m] * torch.tensor([0.15, 1.0, 1.0], device=q.device), octaves=4)
        g = 0.5 + 0.5 * torch.sin(6.2832 * (q[m, 1] * 0.5 + 3.0 * w))
        col[m] = torch.lerp(c1[m], c2[m], (g ** 3)[:, None])
        mod[m] = 1.0 + amp[m] * (n[m] - 0.5)
    m = sel("split")
    if m.any():
        z = loc_n[m, 2]
        col[m] = torch.where((z > 0.0)[:, None], c1[m], c2[m])
        mod[m] = 1.0 + amp[m] * (n[m] - 0.5)
    m = sel("stripes")
    if m.any():
        b = torch.sin(q[m, 2]) > 0.55
        col[m] = torch.where(b[:, None], c2[m], c1[m])
        mod[m] = 1.0 + amp[m] * (n[m] - 0.5)
    m = sel("blotch")
    if m.any():
        b = fbm(q[m] * 0.35 + 5.0, octaves=3)
        k = ((b - 0.52) / 0.06).clamp(0.0, 1.0)
        col[m] = torch.lerp(c1[m], c2[m], k[:, None])
        mod[m] = 1.0 + amp[m] * (n[m] - 0.5)
    m = sel("tiles")
    if m.any():
        tq = loc[m, :2] * freq[m, None]                      # tile units
        ti = torch.floor(tq)
        fr = tq - ti
        grout = ((fr < 0.03) | (fr > 0.97)).any(-1)
        shade = 0.92 + 0.16 * _hash(ti[:, 0].long(), ti[:, 1].long(), torch.zeros_like(ti[:, 0]).long())
        col[m] = torch.where(grout[:, None], c2[m], c1[m] * shade[:, None])
        speck = fbm(q[m] * 9.0, octaves=2)
        mod[m] = 1.0 + amp[m] * (speck - 0.5) * 2.0
    m = sel("brushed")
    if m.any():
        b = fbm(q[m] * torch.tensor([0.05, 4.0, 4.0], device=q.device), octaves=2)
        mod[m] = 1.0 + amp[m] * (b - 0.5) * 2.0
    return (col * mod[:, None]).clamp(0.0, 1.0)


def specular_params(scene, prim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(exponent, strength, metallic) per hit from roughness / metallic."""
    r = scene.roughness[prim].clamp(0.02, 1.0)
    exponent = (2.0 / (r * r) - 2.0).clamp(4.0, 400.0)
    strength = 0.03 + 0.55 * (1.0 - r) ** 2
    return exponent, strength, scene.metallic[prim]
