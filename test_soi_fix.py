# test_soi_fix.py - Test SOI with full library search
import numpy as np

wl_lib = np.arange(400, 1001, dtype=float)

# Load raw SOI spectra (all 500K)
print("Loading SOI spectra (500K × 601)...")
spec_soi = np.fromfile(r'D:\kd_forest_v2\bench_data\spec_soi.bin', dtype=np.float32)
spec_soi = spec_soi.reshape(-1, 601).astype(np.float64)
thick_top = np.fromfile(r'D:\kd_forest_v2\bench_data\thick_soi.bin', dtype=np.float32)
thick_box = np.fromfile(r'D:\kd_forest_v2\bench_data\thick_soi_box.bin', dtype=np.float32)
print(f"  spec_soi: {spec_soi.shape}")
print(f"  top Si range: [{thick_top.min():.0f}, {thick_top.max():.0f}]")
print(f"  BOX range: [{thick_box.min():.0f}, {thick_box.max():.0f}]")

# L2 normalize library
norms = np.linalg.norm(spec_soi, axis=1, keepdims=True)
norms[norms == 0] = 1
spec_soi_n = spec_soi / norms

# Real SOI files
soi_files = [
    ('0.5umSOI.csv', 500, 0.5),
    ('1umSOI.csv', 1000, 1.0),
    ('2umSOI.csv', 2000, 2.0),
    ('3umSOI.csv', 3000, 3.0),
    ('4umSOI.csv', 4000, 4.0),
    ('5umSOI.csv', 5000, 5.0),
    ('6umSOI.csv', 6000, 6.0),
    ('7umSOI.csv', 7000, 7.0),
]

print(f"\n{'File':20s} {'GT_top':>8s} {'Best_top':>10s} {'Best_BOX':>10s} {'Dist':>10s} {'OK':>4s}")
print('-'*60)

# Also load other materials for comparison
mats = {'ox': r'D:\kd_forest_v2\bench_data\spec_ox.bin',
        'sin': r'D:\kd_forest_v2\bench_data\spec_sin.bin',
        'cauthy': r'D:\kd_forest_v2\bench_data\spec_cauthy.bin'}

mat_specs = {}
for mn, fp in mats.items():
    s = np.fromfile(fp, dtype=np.float32).reshape(-1, 601).astype(np.float64)
    n = np.linalg.norm(s, axis=1, keepdims=True)
    n[n==0]=1; s = s/n
    mat_specs[mn] = s
    print(f"  {mn}: {s.shape}")

for fname, gt_top, gt_um in soi_files:
    fp = rf'D:\kd_forest_v2\test_data\CE\SOI\{fname}'
    d = np.loadtxt(fp, delimiter=',', skiprows=2)
    I = np.interp(wl_lib, d[:,0], d[:,1])
    I = I / np.linalg.norm(I)
    
    # 1. Search SOI library (601D L2)
    diffs = spec_soi_n - I.reshape(1, -1)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    best_soi_idx = dists.argmin()
    best_soi_d = dists[best_soi_idx]
    
    print(f"\n{fname:20s} {gt_top:>8.0f}nm  "
          f"SOI→ top={thick_top[best_soi_idx]:.0f}nm "
          f"BOX={thick_box[best_soi_idx]:.0f}nm "
          f"dist={best_soi_d:.6f}")
    
    # 2. Compare to other materials (sample every 1000)
    for mn in ['ox', 'sin', 'cauthy']:
        s = mat_specs[mn]
        diffs = s[::1000] - I.reshape(1, -1)
        d = np.sqrt(np.sum(diffs**2, axis=1))
        print(f"  → {mn:6s}: min_dist={d.min():.6f}")
    
    # Check: is BOX thickness of best match reasonable?
    # Typical SOI BOX is 145nm, 200nm, 400nm
    box_at_best = thick_box[best_soi_idx]
    print(f"  Best SOI match: top={thick_top[best_soi_idx]:.0f}nm, BOX={box_at_best:.0f}nm")

# Also try: fix BOX=145nm and search
print("\n\n=== Fixed BOX=145nm search ===")
BOX_FIXED = 200  # try common values
idx_box = np.where(np.abs(thick_box - BOX_FIXED) < 1)[0]
print(f"  BOX={BOX_FIXED}nm: {len(idx_box)} spectra in library")
if len(idx_box) > 0:
    soi_box_fixed = spec_soi_n[idx_box]
    thick_box_fixed = thick_top[idx_box]
    
    for fname, gt_top, gt_um in soi_files:
        fp = rf'D:\kd_forest_v2\test_data\CE\SOI\{fname}'
        d = np.loadtxt(fp, delimiter=',', skiprows=2)
        I = np.interp(wl_lib, d[:,0], d[:,1])
        I = I / np.linalg.norm(I)
        
        diffs = soi_box_fixed - I.reshape(1, -1)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        best_idx = dists.argmin()
        print(f"  {fname:20s} top={thick_box_fixed[best_idx]:.0f}nm dist={dists.min():.6f}")

# Check: scan BOX from 100 to 500 to find best per file
print("\n\n=== BOX scan for each SOI file ===")
for fname, gt_top, gt_um in soi_files:
    fp = rf'D:\kd_forest_v2\test_data\CE\SOI\{fname}'
    d = np.loadtxt(fp, delimiter=',', skiprows=2)
    I = np.interp(wl_lib, d[:,0], d[:,1])
    I = I / np.linalg.norm(I)
    
    best_overall_d = float('inf')
    best_overall_box = 0
    best_overall_top = 0
    
    for box_target in [50, 100, 145, 200, 300, 400, 500, 1000, 2000]:
        idx_b = np.where(np.abs(thick_box - box_target) < 2)[0]
        if len(idx_b) == 0: continue
        s = spec_soi_n[idx_b]
        diffs = s - I.reshape(1, -1)
        d = np.sqrt(np.sum(diffs**2, axis=1))
        min_idx = idx_b[d.argmin()]
        if d.min() < best_overall_d:
            best_overall_d = d.min()
            best_overall_box = thick_box[min_idx]
            best_overall_top = thick_top[min_idx]
    
    print(f"  {fname:20s} GT={gt_top:5.0f}nm → "
          f"pred top={best_overall_top:.0f}nm BOX={best_overall_box:.0f}nm dist={best_overall_d:.6f}")
