# test_raw_601d.py - Test real data classification with 601D raw spectra
# Earlier (06-11 11:06) full BF 601D L2 search gave 100% - let's verify
import numpy as np, os

wl_lib = np.arange(400, 1001, dtype=float)

# Load a sampling of raw library spectra for each material
# spec_*.bin is (500000, 601) float32 = 1.2GB each
# Sample every 500 → 1000 spectra per material
mats = ['ox','sin','soi','cauthy']
print("Loading library spectra (sampled every 500th)...")
lib_raw = {}
lib_thick = {}
for mn in mats:
    s = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\spec_{mn}.bin', dtype=np.float32)
    s = s.reshape(-1, 601)
    t = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\thick_{mn}.bin', dtype=np.float32)
    stride = 500
    lib_raw[mn] = s[::stride].astype(np.float64)  # 1000 specs
    lib_thick[mn] = t[::stride]
    print(f"  {mn}: {lib_raw[mn].shape}")

# Normalize each library spectrum to L2
for mn in mats:
    norms = np.linalg.norm(lib_raw[mn], axis=1, keepdims=True)
    norms[norms == 0] = 1
    lib_raw[mn] = lib_raw[mn] / norms

# Load all real CSV files
real_files = []
for dirname in ['OX','SIN','SOI','CAUTYONGLASS','POLY']:
    base = rf'D:\kd_forest_v2\test_data\CE\{dirname}'
    if not os.path.isdir(base): continue
    gt = dirname[:4].lower()
    if gt == 'caut': gt = 'cauthy'
    for fn in sorted(os.listdir(base)):
        if not fn.endswith('.csv'): continue
        d = np.loadtxt(os.path.join(base, fn), delimiter=',', skiprows=2)
        I = np.interp(wl_lib, d[:,0], d[:,1])
        I = I / np.linalg.norm(I)
        real_files.append((os.path.basename(fn), gt, I, 0))

print(f"\nLoaded {len(real_files)} real files")

# BF search in raw 601D space
print(f"\n{'File':30s}  {'GT':8s}  {'Pred':8s}  {'Dist':10s}  {'Thick':>8s}  {'OK':>4s}")
print('-'*70)

correct = 0
for fname, gt, I, _ in real_files:
    best_mn = None
    best_d = float('inf')
    best_t = 0
    
    for mn in mats:
        # L2 distances in 601D space
        diffs = lib_raw[mn] - I.reshape(1, -1)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_idx = dists.argmin()
        if dists[min_idx] < best_d:
            best_d = dists[min_idx]
            best_mn = mn
            best_t = lib_thick[mn][min_idx]
    
    ok = 'OK' if best_mn == gt else 'XX'
    if ok == 'OK': correct += 1
    print(f"{fname:30s}  {gt:8s}  {best_mn:8s}  {best_d:>10.6f}  {best_t:>8.1f}  {ok:>4s}")

print(f"\n{'='*50}")
print(f"Total: {correct}/{len(real_files)} = {100*correct/len(real_files):.1f}%")

# Also check: what if we include POLY library?
# POLY was in 4-material library... let me check
# Actually spec_poly.bin exists
print("\nChecking if spec_poly.bin exists:", os.path.exists(r'D:\kd_forest_v2\bench_data\spec_poly.bin'))
