"""Test multiple routing strategies on real CSV"""
import numpy as np
BD = r'D:\kd_forest_v2_gh\src\bench_data'
NW, ND = 601, 50
MNS = ['ox', 'sin', 'soi', 'cauthy']

# Real CSV
rs = np.fromfile('real_csv/real_specs.bin', dtype=np.float32).reshape(-1, NW)
rlb = np.fromfile('real_csv/real_labels.bin', dtype=np.int32)
print(f"Real queries: {len(rs)}")

# Global PCA from bench_data
gmean = np.fromfile(f'{BD}/pca_mean_601.bin', dtype=np.float32)
gcomp = np.fromfile(f'{BD}/pca_comp_50x601.bin', dtype=np.float32).reshape(ND, NW)

# Load per-material reference sets
libs = {}
for mi, m in enumerate(MNS):
    libs[m] = np.fromfile(f'{BD}/lib_{m}_n.bin', dtype=np.float32).reshape(-1, NW)

def test_strategy(name, fn, use_libs=True):
    ok = 0
    for i in range(len(rs)):
        pred = fn(rs[i], i)
        if pred == rlb[i]:
            ok += 1
    print(f"  {name:>35s}: {ok}/{len(rs)} ({100*ok/len(rs):.1f}%)")

# 1. PCA-50D + 6K route set (baseline)
rp, rlab = [], []
for mi, m in enumerate(MNS):
    d = libs[m][:6000]
    p = ((d - gmean) @ gcomp.T).astype(np.float32)
    n = np.sqrt((p**2).sum(axis=1, keepdims=True)) + 1e-12
    p /= n
    rp.append(p)
    rlab.append(np.full(6000, mi, dtype=np.int32))
rp = np.vstack(rp).astype(np.float32)
rlab = np.hstack(rlab)
def route_pca6k(q, idx):
    proj = ((q - gmean) @ gcomp.T).astype(np.float32)
    proj /= np.sqrt((proj**2).sum()) + 1e-12
    return rlab[((rp - proj)**2).sum(axis=1).argmin()]
test_strategy("1. PCA-50D + 6K route (baseline)", route_pca6k)

# 2. Derivative spectra PCA routing
dgmean = np.gradient(gmean)[-500:]
# For derivative, compute PCA on derivative of library
deriv_libs = {}
for m in MNS:
    deriv_libs[m] = np.gradient(libs[m][:20000], axis=1).astype(np.float64)
X_deriv = np.vstack([deriv_libs[m] for m in MNS])
dmean = X_deriv.mean(axis=0).astype(np.float32)
Xc = X_deriv - dmean
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
dcomp = Vt[:50].astype(np.float32)

drp, drlab = [], []
for mi, m in enumerate(MNS):
    d = deriv_libs[m][:6000]
    p = ((d - dmean) @ dcomp.T).astype(np.float32)
    n = np.sqrt((p**2).sum(axis=1, keepdims=True)) + 1e-12
    p /= n
    drp.append(p)
    drlab.append(np.full(6000, mi, dtype=np.int32))
drp = np.vstack(drp).astype(np.float32)
drlab = np.hstack(drlab)
def route_deriv(q, idx):
    dq = np.gradient(q).astype(np.float32)
    proj = ((dq - dmean) @ dcomp.T).astype(np.float32)
    proj /= np.sqrt((proj**2).sum()) + 1e-12
    return drlab[((drp - proj)**2).sum(axis=1).argmin()]
test_strategy("2. Derivative PCA-50D + 6K route", route_deriv)

# 3. Centroid (mean spectrum) nearest neighbor
centroids = []
for m in MNS:
    centroids.append(libs[m][:20000].mean(axis=0).astype(np.float32))
centroids = np.array(centroids)
def route_centroid(q, idx):
    d2 = ((centroids - q)**2).sum(axis=1)
    return np.argmin(d2)
test_strategy("3. Centroid NN (601D mean)", route_centroid)

# 4. Centroid with derivative
dcentroids = np.array([np.gradient(centroids[i]) for i in range(4)]).astype(np.float32)
def route_dcentroid(q, idx):
    dq = np.gradient(q).astype(np.float32)
    d2 = ((dcentroids - dq)**2).sum(axis=1)
    return np.argmin(d2)
test_strategy("4. Derivative centroid NN", route_dcentroid)

# 5. PCA reconstruction error per material
per_mean = {}; per_comp = {}
for mi, m in enumerate(MNS):
    X = libs[m][:20000].astype(np.float64)
    mean = X.mean(axis=0).astype(np.float32)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    per_mean[m] = mean
    per_comp[m] = Vt[:50].astype(np.float32)
def route_recon_err(q, idx):
    best_m, best_err = 0, 1e30
    for mi, m in enumerate(MNS):
        proj = ((q - per_mean[m]) @ per_comp[m].T).astype(np.float32)
        recon = (proj @ per_comp[m]) + per_mean[m]
        err = ((recon - q)**2).sum()
        if err < best_err:
            best_err = err
            best_m = mi
    return best_m
test_strategy("5. Per-material PCA-50D recon error", route_recon_err)

# 6. 601D L2 to nearest library sample (1K/subset per material)
subsets = {}
for mi, m in enumerate(MNS):
    subsets[m] = libs[m][:2000]  # uniform sampling from thin end
def route_1k(q, idx):
    best_m, best_d = 0, 1e30
    for mi, m in enumerate(MNS):
        d2 = ((subsets[m] - q)**2).sum(axis=1).min()
        if d2 < best_d:
            best_d = d2
            best_m = mi
    return best_m
test_strategy("6. 601D L2 + 2K/材料 BF route", route_1k)

# 7. 601D L2 + PCA-dense 2K (first 2K has NO thick samples, try every 250th from first 500K)
subsets2 = {}
for mi, m in enumerate(MNS):
    subsets2[m] = libs[m][::250][:2000]  # stride=250 for uniform thickness coverage
def route_2k_stride(q, idx):
    best_m, best_d = 0, 1e30
    for mi, m in enumerate(MNS):
        d2 = ((subsets2[m] - q)**2).sum(axis=1).min()
        if d2 < best_d:
            best_d = d2
            best_m = mi
    return best_m
test_strategy("7. 601D L2 + stride-250×2K route", route_2k_stride)

# 8. Ratio spectroscopy (divide by reference med)
# For each material, use median spectrum; compute ratio; PCA on ratios
print()
print("=== Deeper analysis: which materials get confused? ===")
for mi, m in enumerate(MNS):
    mask = rlb == mi
    for mj, m2 in enumerate(MNS):
        if mi == mj: continue
        wr = sum(1 for i in range(mask.sum()) 
                 if ((centroids[mj] - rs[mask][i])**2).sum() < ((centroids[mi] - rs[mask][i])**2).sum())
        if wr:
            print(f"  {m}: {wr}/{(mask).sum()} misclassified as {m2}")
