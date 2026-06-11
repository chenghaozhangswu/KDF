import numpy as np, time
np.random.seed(42)
NW = 601; NQ_PER = 500
MAT_NAMES = ['ox','sin','soi','cauthy']
BD = 'bench_data'

print("=== Loading libraries (10K/material) ===")
lib = {}; thick = {}
for name in MAT_NAMES:
    raw = np.fromfile(f'{BD}/spec_{name}.bin', dtype=np.float32).reshape(-1, NW)
    t = np.fromfile(f'{BD}/thick_{name}.bin', dtype=np.float32)
    step = max(1, raw.shape[0] // 10000)
    lib[name] = raw[::step].copy()
    thick[name] = t[::step].copy()
    print(f"  {name}: {lib[name].shape[0]}")

# L2-normalize
def ln(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
lib_n = {m: ln(lib[m]) for m in MAT_NAMES}

# Build queries
print("\n=== 2000 queries (500x4, 0-5% noise) ===")
queries = []; tmat = []; tthick = []
for m, name in enumerate(MAT_NAMES):
    raw = np.fromfile(f'{BD}/spec_{name}.bin', dtype=np.float32).reshape(-1, NW)
    t = np.fromfile(f'{BD}/thick_{name}.bin', dtype=np.float32)
    step = max(1, raw.shape[0] // NQ_PER)
    for i in range(NQ_PER):
        idx = min(i*step + step//2, raw.shape[0]-1)
        s = raw[idx].astype(np.float64)
        nl = np.random.uniform(0, 0.05)
        queries.append(s + s * np.random.normal(0, nl, NW).astype(np.float64))
        tmat.append(m); tthick.append(t[idx])
queries = np.array(queries, dtype=np.float64)
tmat = np.array(tmat); tthick = np.array(tthick)
q_n = ln(queries)
print(f"  Shape: {queries.shape}")

# Step 1: BF 50D Router
print("\n=== Step 1: BF 50D Router ===")
pca_mean = np.fromfile(f'{BD}/pca_mean_601.bin', dtype=np.float32).astype(np.float64)
cr = np.fromfile(f'{BD}/pca_comp_50x601.bin', dtype=np.float32).astype(np.float64).reshape(50, 601).T
q_pca = (q_n - pca_mean) @ cr
q_pca_n = q_pca / (np.linalg.norm(q_pca, axis=1, keepdims=True) + 1e-12)
rp = np.fromfile(f'{BD}/route_pca50d.bin', dtype=np.float32).astype(np.float64).reshape(-1, 50)
rl = np.fromfile(f'{BD}/route_labels.bin', dtype=np.int32)
rn = rp / (np.linalg.norm(rp, axis=1, keepdims=True) + 1e-12)
print(f"  Route: {rn.shape}")

t0 = time.perf_counter()
pred = np.array([rl[np.argmin(np.linalg.norm(rn - q_pca_n[i], axis=1))] for i in range(2000)])
rt = (time.perf_counter()-t0)/2000*1e6
ra = np.mean(pred == tmat)*100
print(f"  Acc: {ra:.1f}%  Lat: {rt:.0f} us/q")

# Per-material PCA via SVD (much faster than eigendecomposition)
print("\n=== Per-material PCA (SVD) ===")
from scipy.spatial import KDTree
n_comp = 100
pca = {}; kdt50 = {}; kdt100 = {}
for name in MAT_NAMES:
    d = lib_n[name].copy()
    if d.dtype != np.float64: d = d.astype(np.float64)
    mean = np.mean(d, axis=0)
    cent = d - mean
    # SVD is faster and more stable for n_samples >> n_features
    U, S, Vt = np.linalg.svd(cent, full_matrices=False)
    comps = Vt.T  # (601, n_components)
    pca[name] = (mean, comps.astype(np.float64))
    # Build KDTs
    d50 = (d - mean) @ comps[:, :50]
    kdt50[name] = KDTree(d50)
    d100 = (d - mean) @ comps[:, :100]
    kdt100[name] = KDTree(d100)
    print(f"  {name}: SVD done, KDT built ({d50.shape[0]} pts)")

# Benchmark
print("\n" + "="*65)
print("Benchmark: 2000 queries x 4 methods")
print("="*65)
nq = 2000; K = 50
res = {}

# 1) BF 601D
print("\n--- [1/4] BF-601D ---")
ok=0; tt=0.0
for i in range(nq):
    mn = MAT_NAMES[pred[i]]
    t0 = time.perf_counter()
    d = np.linalg.norm(lib_n[mn] - q_n[i].astype(np.float64), axis=1)
    bi = np.argmin(d)
    tt += time.perf_counter()-t0
    if abs(thick[mn][bi] - tthick[i]) <= 1.0: ok += 1
    if (i+1)%500==0: print(f"  {i+1}/{nq} P1nm={ok/(i+1)*100:.1f}% lat={tt*1e6/(i+1):.0f} us/q")
res['BF-601D'] = (ok/nq*100, tt*1e6/nq)

# 2) KDT-50D
print("\n--- [2/4] KDT-50D ---")
ok=0; tt=0.0
for i in range(nq):
    mn = MAT_NAMES[pred[i]]
    mean, comps = pca[mn]
    q50 = (q_n[i].astype(np.float64) - mean) @ comps[:, :50]
    q50n = q50 / (np.linalg.norm(q50)+1e-12)
    t0 = time.perf_counter()
    _, idx = kdt50[mn].query(q50n[np.newaxis,:], k=1)
    tt += time.perf_counter()-t0
    bi = idx[0]
    if abs(thick[mn][bi] - tthick[i]) <= 1.0: ok += 1
    if (i+1)%500==0: print(f"  {i+1}/{nq} P1nm={ok/(i+1)*100:.1f}% lat={tt*1e6/(i+1):.0f} us/q")
res['KDT-50D'] = (ok/nq*100, tt*1e6/nq)

# 3) KDT-100D
print("\n--- [3/4] KDT-100D ---")
ok=0; tt=0.0
for i in range(nq):
    mn = MAT_NAMES[pred[i]]
    mean, comps = pca[mn]
    q100 = (q_n[i].astype(np.float64) - mean) @ comps[:, :100]
    q100n = q100 / (np.linalg.norm(q100)+1e-12)
    t0 = time.perf_counter()
    _, idx = kdt100[mn].query(q100n[np.newaxis,:], k=1)
    tt += time.perf_counter()-t0
    bi = idx[0]
    if abs(thick[mn][bi] - tthick[i]) <= 1.0: ok += 1
    if (i+1)%500==0: print(f"  {i+1}/{nq} P1nm={ok/(i+1)*100:.1f}% lat={tt*1e6/(i+1):.0f} us/q")
res['KDT-100D'] = (ok/nq*100, tt*1e6/nq)

# 4) KDF: KDT-50D -> 601D L2 rerank top K
print(f"\n--- [4/4] KDF (KDT-50D -> 601D rerank K={K}) ---")
ok=0; tt=0.0
for i in range(nq):
    mn = MAT_NAMES[pred[i]]
    mean, comps = pca[mn]
    q50 = (q_n[i].astype(np.float64) - mean) @ comps[:, :50]
    q50n = q50 / (np.linalg.norm(q50)+1e-12)
    t0 = time.perf_counter()
    _, idx = kdt50[mn].query(q50n[np.newaxis,:], k=K)
    cand = idx[0]
    rr = np.linalg.norm(lib_n[mn][cand] - q_n[i].astype(np.float64), axis=1)
    best = cand[np.argmin(rr)]
    tt += time.perf_counter()-t0
    if abs(thick[mn][best] - tthick[i]) <= 1.0: ok += 1
    if (i+1)%500==0: print(f"  {i+1}/{nq} P1nm={ok/(i+1)*100:.1f}% lat={tt*1e6/(i+1):.0f} us/q")
res['KDF-50/50'] = (ok/nq*100, tt*1e6/nq)

# Summary
print("\n" + "="*65)
print(f"{'Method':<22} {'P1nm':>8} {'Lat (us/q)':>12}")
print("-"*45)
for name, (a, l) in sorted(res.items(), key=lambda x: -x[1][0]):
    print(f"{name:<22} {a:>7.1f}% {l:>10.0f}")
print("="*65)
print(f"Router: BF-50D-OMP ({ra:.1f}%) @ {rt:.0f} us/q")
print(f"Lib size: 10K/material | Queries: 2000 | Noise: 0-5%")
