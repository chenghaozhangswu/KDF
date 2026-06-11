"""PCA 50D route forest: fast classification benchmark"""
import numpy as np
from scipy.spatial import KDTree
import time, os, glob

NW = 601
NMATS = 4
MAT_NAMES = ['ox','sin','soi','cauthy']
MAT_DIRS = ['OX','SIN','SOI','CAUTYONGLASS']
BENCH = r'D:\kd_forest_v2\bench_data'
TEST = r'D:\kd_forest_v2\test_data\CE'

def load_bin(path, cols=NW):
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, cols)

def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms

def pca_transform(X, comps, mean):
    return (X - mean) @ comps.T

def bf_classify(q, rdata):
    """Brute-force L2 on 50D"""
    dists = np.linalg.norm(rdata - q, axis=1)
    return np.argmin(dists)

print("=== PCA 50D Route Forest ===")

# Load PCA
print("\n--- Loading PCA / building route set ---")
pca_mean = np.load(os.path.join(BENCH, 'pca_mean.npy')).astype(np.float32)
pca_comp = np.load(os.path.join(BENCH, 'pca_comp.npy')).astype(np.float32)[:50]

rdata, rlabel = [], []
for m in range(NMATS):
    raw = load_bin(os.path.join(BENCH, f'spec_{MAT_NAMES[m]}.bin'))
    if m == 2:  # SOI
        nr, nc = 500, 1000
        sr, sc = 2, 5
        for r in range(0, nr, sr):
            for c in range(0, nc, sc):
                rdata.append(raw[r * nc + c])
                rlabel.append(m)
    else:
        for i in range(0, raw.shape[0], 100):
            rdata.append(raw[i])
            rlabel.append(m)

rdata = np.array(rdata, dtype=np.float32)  # 65K x 601
rlabel = np.array(rlabel, dtype=np.int32)
rdata_norm = l2norm(rdata)
rdata_pca = pca_transform(rdata_norm, pca_comp, pca_mean)  # 65K x 50D
print(f"  route set: {rdata_pca.shape[0]} pts x {rdata_pca.shape[1]}D = {rdata_pca.nbytes/1024:.0f} KB")

# KDTree
t0 = time.perf_counter()
tree = KDTree(rdata_pca, leafsize=16)
print(f"  KDT build: {(time.perf_counter()-t0)*1000:.1f} ms")

# Simulation queries
print("\n--- Simulation query ---")
qdata, qlabel = [], []
for m in range(NMATS):
    raw = load_bin(os.path.join(BENCH, f'spec_{MAT_NAMES[m]}.bin'))
    step = max(1, raw.shape[0] // 1500)
    for i in range(0, raw.shape[0], step):
        qdata.append(raw[i])
        qlabel.append(m)

qdata = np.array(qdata, dtype=np.float32)
qlabel = np.array(qlabel, dtype=np.int32)
qdata_pca = pca_transform(l2norm(qdata), pca_comp, pca_mean)
nq = qdata_pca.shape[0]
print(f"  queries: {nq}")

# KDT classify (batch)
t0 = time.perf_counter()
_, idx = tree.query(qdata_pca, k=1)
pred_kdt = rlabel[idx.flatten()]
dt_kdt = (time.perf_counter() - t0) * 1e6 / nq

# BF classify (first 500 queries for timing)
t0 = time.perf_counter()
pred_bf = np.array([bf_classify(q, rdata_pca) for q in qdata_pca[:500]])
dt_bf = (time.perf_counter() - t0) * 1e6 / 500

# KDT accuracy
print(f"\n--- KDT classification ---")
for m in range(NMATS):
    mask = qlabel == m
    ok = (pred_kdt[mask] == m).sum()
    print(f"  {MAT_NAMES[m]}: {ok}/{mask.sum()} = {100*ok/mask.sum():.1f}%")
total_kdt = (pred_kdt == qlabel).sum()
print(f"  TOTAL: {total_kdt}/{nq} = {100*total_kdt/nq:.1f}%")
print(f"  KDT latency: {dt_kdt:.1f} us/q")

# BF accuracy (all)
pred_bf_all = rlabel[np.argmin(np.sum((rdata_pca[None,:,:] - qdata_pca[:,None,:])**2, axis=2), axis=1)]
total_bf = (pred_bf_all == qlabel).sum()
print(f"\n--- BF classification ---")
for m in range(NMATS):
    mask = qlabel == m
    ok = (pred_bf_all[mask] == m).sum()
    print(f"  {MAT_NAMES[m]}: {ok}/{mask.sum()} = {100*ok/mask.sum():.1f}%")
print(f"  TOTAL: {total_bf}/{nq} = {100*total_bf/nq:.1f}%")
print(f"  BF latency (first 500): {dt_bf:.1f} us/q")

# Real CSV
print("\n--- Real CSV ---")
csv_ok = 0
samples = []
for m in range(NMATS):
    for fn in glob.glob(os.path.join(TEST, MAT_DIRS[m], '*.csv')):
        d = np.loadtxt(fn, delimiter=',', skiprows=2)
        t = np.linspace(400, 1000, NW)
        spec = np.interp(t, d[:,0], d[:,1], left=d[0,1], right=d[-1,1]).astype(np.float32)
        samples.append((os.path.basename(fn), m, spec))

for fn, gt, spec in samples:
    spec_pca = pca_transform(l2norm(spec.reshape(1,-1)), pca_comp, pca_mean)
    d = np.linalg.norm(rdata_pca - spec_pca[0], axis=1)
    mat = rlabel[np.argmin(d)]
    csv_ok += (mat == gt)
    print(f"  {fn:28s} -> {MAT_NAMES[mat]} (gt={MAT_NAMES[gt]}) {'OK' if mat==gt else 'XX'}")

print(f"  CSV total: {csv_ok}/{len(samples)} = {100*csv_ok/len(samples):.1f}%")
print(f"\n===== DONE =====")
