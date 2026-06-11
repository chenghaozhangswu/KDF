# debug_real_vs_lib.py - Real vs library spectral analysis
import numpy as np, os, glob

# Load real OX 100nm CSV
data = np.loadtxt(r'D:\kd_forest_v2\test_data\CE\OX\OX100nm.csv', delimiter=',', skiprows=2)
wl_real, I_real = data[:,0], data[:,1]

# Load one raw library spectrum (spec_ox.bin has 500K × 601)
spec_ox = np.fromfile(r'D:\kd_forest_v2\bench_data\spec_ox.bin', dtype=np.float32).reshape(-1, 601)
print(f"spec_ox: {spec_ox.shape}")  # (500000, 601)

# Library WL is 400-1000nm
wl_lib = np.arange(400, 1001, dtype=float)
print(f"Library WL: {wl_lib[0]:.0f}-{wl_lib[-1]:.0f}nm, {len(wl_lib)} pts")
print(f"Real CSV WL: {wl_real[0]:.0f}-{wl_real[-1]:.0f}nm, {len(wl_real)} pts")

# Interpolate real data to library grid
I_real_int = np.interp(wl_lib, wl_real, I_real)
I_real_norm = I_real_int / np.linalg.norm(I_real_int)

# Compare to library spectra at various thicknesses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Plot 1: Raw spectra comparison (no normalization)
ax = axes[0,0]
ax.plot(wl_lib, spec_ox[0], 'b-', alpha=0.5, label='Lib OX thick=1nm')
thick_file = np.fromfile(r'D:\kd_forest_v2\bench_data\thick_ox.bin', dtype=np.float32)
for target_t in [100, 500, 1000]:
    idx = np.argmin(np.abs(thick_file - target_t))
    ax.plot(wl_lib, spec_ox[idx], alpha=0.7, label=f'Lib OX {thick_file[idx]:.0f}nm')
ax.plot(wl_lib, I_real_norm, 'k-', linewidth=2, label='Real OX 100nm (L2 norm)')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity (L2 norm)')
ax.set_title('Spectra after L2 normalization')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Plot 2: Raw without normalization
ax = axes[0,1]
ax.plot(wl_real, I_real, 'k-', linewidth=2, label=f'Real OX 100nm')
for target_t in [100, 500, 1000]:
    idx = np.argmin(np.abs(thick_file - target_t))
    ax.plot(wl_lib, spec_ox[idx], alpha=0.7, label=f'Lib OX {thick_file[idx]:.0f}nm (raw)')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity (raw)')
ax.set_title('Raw spectra (no normalization)')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Plot 3: Spectral difference (detrending)
ax = axes[1,0]
# Real - interpolated polynomial fit to remove envelope
coeff = np.polyfit(wl_lib, I_real_int, 15)
envelope = np.polyval(coeff, wl_lib)
I_real_detrended = I_real_int - envelope
I_real_det_norm = I_real_detrended / np.linalg.norm(I_real_detrended)

# Library spectra detrended
for target_t in [100, 500]:
    idx = np.argmin(np.abs(thick_file - target_t))
    s = spec_ox[idx].astype(np.float64)
    coeff_l = np.polyfit(wl_lib, s, 15)
    env_l = np.polyval(coeff_l, wl_lib)
    s_det = s - env_l
    s_det = s_det / np.linalg.norm(s_det)
    ax.plot(wl_lib, s_det, '--', alpha=0.7, label=f'Lib OX {thick_file[idx]:.0f}nm detrended')

ax.plot(wl_lib, I_real_det_norm, 'k-', linewidth=2, label='Real OX 100nm detrended')
ax.set_xlabel('Wavelength (nm)')
ax.set_title('Detrended (polyfit 15th order)')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Plot 4: All materials for one real OX spectrum - BF distances
ax = axes[1,1]
pca_mean = np.load(r'D:\kd_forest_v2\bench_data\pca_mean.npy').astype(np.float64)
pca_comp = np.load(r'D:\kd_forest_v2\bench_data\pca_comp.npy').astype(np.float64)

pca_real = (I_real_norm - pca_mean) @ pca_comp[:, :50]

mats = ['ox','sin','soi','cauthy']
colors = ['b','g','r','c']
width = 0.18
x = np.arange(len(mats))
for i, mn in enumerate(mats):
    lib = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\lib_{mn}_pca50d.bin', dtype=np.float32).reshape(-1, 50)
    dists = np.sqrt(np.sum((lib - pca_real.reshape(1, -1))**2, axis=1))
    min_idx = dists.argmin()
    ax.bar(x[i], dists.min(), width, color=colors[i], alpha=0.7, 
           label=f'{mn}: min={dists.min():.5f} @ idx={min_idx}')
    # Show thick at min idx
    thick_file_mat = f'D:\\kd_forest_v2\\bench_data\\thick_{mn}.bin'
    if os.path.exists(thick_file_mat):
        t = np.fromfile(thick_file_mat, dtype=np.float32)[min_idx]
        print(f"  Real OX100nm -> {mn}: min_dist={dists.min():.6f} @ thick={t:.1f}nm")

ax.set_xticks(x)
ax.set_xticklabels(mats)
ax.set_ylabel('PCA 50D L2 distance')
ax.set_title('Real OX 100nm: nearest in each material')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 5: Real spectra for OX, SIN, SOI
ax = axes[2,0]
for fn, lbl, clr in [
    (r'D:\kd_forest_v2\test_data\CE\OX\OX100nm.csv', 'OX 100nm', 'b'),
    (r'D:\kd_forest_v2\test_data\CE\SIN\SIN150nm.csv', 'SIN 150nm', 'g'),
    (r'D:\kd_forest_v2\test_data\CE\SOI\2umSOI.csv', 'SOI 2um', 'r'),
    (r'D:\kd_forest_v2\test_data\CE\CAUTYONGLASS\Cauthyonglass500nm.csv', 'CAUTHY 500nm', 'c'),
    (r'D:\kd_forest_v2\test_data\CE\POLY\Poly500nm.csv', 'POLY 500nm', 'm'),
]:
    d = np.loadtxt(fn, delimiter=',', skiprows=2)
    I_int = np.interp(wl_lib, d[:,0], d[:,1])
    I_n = I_int / np.linalg.norm(I_int)
    ax.plot(wl_lib, I_n, color=clr, alpha=0.7, label=lbl)

# Also plot library spectra for comparison
spec_lib = spec_ox.astype(np.float64)
s_n = spec_lib[15000] / np.linalg.norm(spec_lib[15000])
ax.plot(wl_lib, s_n, 'k--', alpha=0.5, label=f'Lib OX {thick_file[15000]:.0f}nm')

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity (L2 norm)')
ax.set_title('All real materials + 1 library ref')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# Plot 6: Amplitude envelope comparison
ax = axes[2,0].twinx()

plt.tight_layout()
plt.savefig(r'D:\kd_forest_v2\real_vs_lib_debug.png', dpi=120)
print(f"\nSaved D:\\kd_forest_v2\\real_vs_lib_debug.png")

# Summary: all 4 materials BF distances for each real file
print(f"\n{'File':30s} {'GT':6s} {'thick':>8s} {'BF_mat':6s} {'BF_dist':>10s} {'BF_thick':>10s}")
real_files = []
for dirname in ['OX','SIN','SOI','CAUTYONGLASS','POLY']:
    base = rf'D:\kd_forest_v2\test_data\CE\{dirname}'
    if not os.path.isdir(base): continue
    for fn in sorted(glob.glob(os.path.join(base, '*.csv'))):
        fname = os.path.basename(fn)
        gt = dirname[:4].lower()
        if gt == 'caut': gt = 'cauthy'
        d = np.loadtxt(fn, delimiter=',', skiprows=2)
        I_int = np.interp(wl_lib, d[:,0], d[:,1])
        I_n = I_int / np.linalg.norm(I_int)
        pca_r = (I_n - pca_mean) @ pca_comp[:, :50]
        
        best_mn, best_d, best_t = None, float('inf'), 0
        for mn in mats:
            lib = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\lib_{mn}_pca50d.bin', dtype=np.float32).reshape(-1, 50)
            dists = np.sqrt(np.sum((lib - pca_r.reshape(1, -1))**2, axis=1))
            min_idx = dists.argmin()
            d_min = dists[min_idx]
            t_file = f'D:\\kd_forest_v2\\bench_data\\thick_{mn}.bin'
            t_val = np.fromfile(t_file, dtype=np.float32)[min_idx] if os.path.exists(t_file) else -1
            if d_min < best_d:
                best_d = d_min; best_mn = mn; best_t = t_val
        
        # Also compute L2 norm on 50D features directly (not PCA)
        I_l2 = I_int / np.linalg.norm(I_int)
        print(f"{fname:30s} {gt:6s} {-1:>8.0f} {best_mn:6s} {best_d:>10.6f} {best_t:>10.1f}")

# Check: are the real CSV CAUTHY actually closest to lib_cauthy?
print(f"\n=== CAUTHY ONLY analysis ===")
import re
for fn in sorted(glob.glob(r'D:\kd_forest_v2\test_data\CE\CAUTYONGLASS\*.csv')):
    d = np.loadtxt(fn, delimiter=',', skiprows=2)
    I_int = np.interp(wl_lib, d[:,0], d[:,1])
    I_n = I_int / np.linalg.norm(I_int)
    pca_r = (I_n - pca_mean) @ pca_comp[:, :50]
    
    for mn in mats:
        lib = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\lib_{mn}_pca50d.bin', dtype=np.float32).reshape(-1, 50)
        dists = np.sqrt(np.sum((lib - pca_r.reshape(1, -1))**2, axis=1))
        d_min = dists.min()
        print(f"  {os.path.basename(fn):30s} -> {mn:6s}: {d_min:.6f}")
    print()
