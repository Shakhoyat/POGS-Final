"""DINOv3 latency with the O5 harness protocol (shared/bench.py: time_model, resolution_for), using the
ungated timm copies of the DINOv3 weights. The study's own loader needs the gated facebook/ snapshot."""
import sys, torch, timm
sys.path.insert(0, "shared")
import bench
torch.backends.cudnn.benchmark = True
h, w = bench.resolution_for("dinov3"); h, w = h // 16 * 16, w // 16 * 16  # harness gives 1260, not divisible by 16
for name in ("vit_small_patch16_dinov3.lvd1689m", "vit_base_patch16_dinov3.lvd1689m"):
    m = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True).cuda().eval()
    for dt in (torch.float32, torch.float16):
        mm = m.to(dt)
        x = torch.rand(1, 3, h, w, device="cuda", dtype=dt)
        with torch.no_grad():
            r = bench.time_model(lambda t: mm.forward_features(t), x, 20, 100)
        print(f"{name} {str(dt).split('.')[-1]} {h}x{w} median {r['median_ms']} ms p95 {r['p95_ms']} ms peak {r['peak_delta_mb']} MB", flush=True)
    del m; torch.cuda.empty_cache()
