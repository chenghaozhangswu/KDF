"""
KD-Forest v2 benchmark — CORRECT setup:
  BF-601D:     1 index (all 1.5M), auto material
  KDT-50D:     3 indices (per material), oracle material
  KDT-100D:    3 indices (per material), oracle material
  HNSW-50D:    3 indices (per material), oracle material
  KDF:         multi-tree + auto routing + 601D rerank
"""
import numpy as np, time, json, faiss, os

DATA = r'D:\kd_forest_v2\bench_data'
OUT = r'D:\kd_forest_v2\results'
os.makedirs(OUT, exist_ok=True)

print("Loading data...", flush=True)

# PCA features
lib_pca50 = np.fromfile(f'{DATA}/lib_pca50_pm.bin', dtype=np.float32).reshape(-1, 50)
lib_thick = np.fromfile(f'{DATA}/lib_thick.bin', dtype=np.float32)
q_pca50 = np.fromfile(f'{DATA}/query_pca50_pm.bin', dtype=np.float32).reshape(-1, 50)
q_thick = np.fromfile(f'{DATA}/query_thick.bin', dtype=np.float32)
q_mat = np.fromfile(f'{DATA}/query_mat.bin', dtype=np.int32)

N, _ = lib_pca50.shape   # 1500000
NQ = q_pca50.shape[0]    # 1500
NLIB = N // 3             # 500000 per material
print(f"Library: {N} points ({NLIB} per mat × 3), Queries: {NQ}", flush=True)

# Split per-material
lib_pca50_m = [lib_pca50[m*NLIB:(m+1)*NLIB] for m in range(3)]
lib_thick_m = [lib_thick[m*NLIB:(m+1)*NLIB] for m in range(3)]

# 601D spectra
print("Loading 601D data...", flush=True)
t0 = time.time()
lib_spec = np.fromfile(f'{DATA}/lib_all_601d.bin', dtype=np.float32).reshape(-1, 601)
q_spec = np.fromfile(f'{DATA}/query_spec.bin', dtype=np.float32).reshape(-1, 601)
print(f"  {time.time()-t0:.1f}s", flush=True)

lib_norm = lib_spec.copy()
lib_norm /= np.linalg.norm(lib_norm, axis=1, keepdims=True)
q_norm = q_spec / np.linalg.norm(q_spec, axis=1, keepdims=True)
del lib_spec, q_spec



K_RERANK = 50

def accuracy(pred, gt):
    return (float(np.mean(np.abs(pred - gt) <= 1.0)),
            float(np.mean(np.abs(pred - gt) <= 5.0)),
            float(np.median(np.abs(pred - gt))))

def rerank(cand_per_q, q_norm, lib_norm_use):
    """Rerank candidates with 601D cosine"""
    pred = np.zeros(len(cand_per_q), dtype=np.float32)
    for qi, cand in enumerate(cand_per_q):
        dots = lib_norm_use[cand] @ q_norm[qi]
        pred[qi] = lib_thick[cand[dots.argmax()]]
    return pred

results = []

# ========================================================================
# 1. BF-601D — full index 1.5M (auto material)
# ========================================================================
print("\n" + "="*60, flush=True)
print("1. BF-601D (full 1.5M, FAISS IndexFlatIP)", flush=True)
print("="*60, flush=True)
t0 = time.time()
bf_idx = faiss.IndexFlatIP(601)
bf_idx.add(lib_norm)
bf_build = (time.time()-t0)*1000
t0 = time.time()
D, I = bf_idx.search(q_norm, 1)
bf_us = (time.time()-t0)/NQ*1e6
bf_pred = lib_thick[I[:,0]]
p1, p5, med = accuracy(bf_pred, q_thick)
r = {'method':'BF-601D','p1nm':p1,'p5nm':p5,'medae':med,
     'build_ms':bf_build,'search_us':bf_us,'accel':1.0}
print(f"  P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  MedAE={med:.2f}nm  "
      f"build={bf_build:.0f}ms  search={bf_us:.0f}us/q  accel=1×", flush=True)
results.append(r)
bf_time_us = bf_us
del bf_idx, D, I

# ========================================================================
# 2. KDT-50D — per-material, oracle, no rerank
# ========================================================================
print("\n" + "="*60, flush=True)
print("2. KDT-50D (per material × 3, no rerank)", flush=True)
print("="*60, flush=True)
pred = np.zeros(NQ, dtype=np.float32)
build_ms = 0
t0_all = time.time()
indices50 = []
for m in range(3):
    t0 = time.time()
    idx = faiss.IndexFlatL2(50)
    idx.add(lib_pca50_m[m])
    build_ms += (time.time()-t0)*1000
    indices50.append(idx)
for qi in range(NQ):
    m = q_mat[qi]
    D, I = indices50[m].search(q_pca50[qi:qi+1], 1)
    pred[qi] = lib_thick_m[m][I[0,0]]
search_us = (time.time()-t0_all)/NQ*1e6
p1, p5, med = accuracy(pred, q_thick)
accel = bf_time_us/search_us
r = {'method':'KDT-50D','p1nm':p1,'p5nm':p5,'medae':med,
     'build_ms':build_ms,'search_us':search_us,'accel':accel}
print(f"  P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  MedAE={med:.2f}nm  "
      f"build={build_ms:.0f}ms  search={search_us:.0f}us/q  accel={accel:.0f}×", flush=True)
results.append(r)

# KDT-50D + rerank
print("\n  KDT-50D + rerank (K=50 → 601D cosine)", flush=True)
pred = np.zeros(NQ, dtype=np.float32)
t0_all = time.time()
for qi in range(NQ):
    m = q_mat[qi]
    D, I = indices50[m].search(q_pca50[qi:qi+1], K_RERANK)
    cand = I[0] + m*NLIB  # convert to global index
    dots = lib_norm[cand] @ q_norm[qi]
    pred[qi] = lib_thick[cand[dots.argmax()]]
search_us = (time.time()-t0_all)/NQ*1e6
p1, p5, med = accuracy(pred, q_thick)
accel = bf_time_us/search_us
r = {'method':'KDT-50D+Rerank','p1nm':p1,'p5nm':p5,'medae':med,
     'build_ms':build_ms,'search_us':search_us,'accel':accel}
print(f"  P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  MedAE={med:.2f}nm  "
      f"build={build_ms:.0f}ms  search={search_us:.0f}us/q  accel={accel:.0f}×", flush=True)
results.append(r)



# ========================================================================
# 4. HNSW-50D — per-material × 3, oracle
# ========================================================================
print("\n" + "="*60, flush=True)
print("4. FAISS HNSW (per material × 3, 50D PCA)", flush=True)
print("="*60, flush=True)

hnsw_configs = [
    (16, 200, 50,  "HNSW-50D(M16)"),
    (16, 200, 200, "HNSW-50D(M16,efS=200)"),
    (8,  200, 50,  "HNSW-50D(M8)"),
]

for M, efC, efS, label in hnsw_configs:
    hnsw_idx = []
    build_ms = 0
    for m in range(3):
        t0 = time.time()
        idx = faiss.IndexHNSWFlat(50, M)
        idx.hnsw.efConstruction = efC
        idx.add(lib_pca50_m[m])
        build_ms += (time.time()-t0)*1000
        hnsw_idx.append(idx)
    
    # No rerank
    pred = np.zeros(NQ, dtype=np.float32)
    t0_all = time.time()
    for qi in range(NQ):
        m = q_mat[qi]
        hnsw_idx[m].hnsw.efSearch = efS
        D, I = hnsw_idx[m].search(q_pca50[qi:qi+1], 1)
        pred[qi] = lib_thick_m[m][I[0,0]]
    su = (time.time()-t0_all)/NQ*1e6
    p1, p5, med = accuracy(pred, q_thick)
    accel = bf_time_us/su
    r = {'method':label,'p1nm':p1,'p5nm':p5,'medae':med,
         'build_ms':build_ms,'search_us':su,'accel':accel}
    print(f"  {label:25s}: P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  MedAE={med:.2f}nm  "
          f"build={build_ms:.0f}ms  search={su:.0f}us/q  accel={accel:.0f}×", flush=True)
    results.append(r)
    
    # + rerank
    pred = np.zeros(NQ, dtype=np.float32)
    t0_all = time.time()
    for qi in range(NQ):
        m = q_mat[qi]
        hnsw_idx[m].hnsw.efSearch = efS
        D, I = hnsw_idx[m].search(q_pca50[qi:qi+1], K_RERANK)
        cand = I[0] + m*NLIB
        dots = lib_norm[cand] @ q_norm[qi]
        pred[qi] = lib_thick[cand[dots.argmax()]]
    su = (time.time()-t0_all)/NQ*1e6
    p1, p5, med = accuracy(pred, q_thick)
    accel = bf_time_us/su
    r = {'method':f'{label}+Rerank','p1nm':p1,'p5nm':p5,'medae':med,
         'build_ms':build_ms,'search_us':su,'accel':accel}
    print(f"  {label+'+Rerank':25s}: P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  MedAE={med:.2f}nm  "
          f"build={build_ms:.0f}ms  search={su:.0f}us/q  accel={accel:.0f}×", flush=True)
    results.append(r)
    del hnsw_idx

# ========================================================================
# 5. KDF — auto routing via multi-tree + 601D rerank
# ========================================================================
print("\n" + "="*60, flush=True)
print("5. KDF (auto routing: 20 trees, 50D PCA, K=50, 601D rerank)", flush=True)
print("="*60, flush=True)

# Build 20 KD-Trees on all 1.5M points, each using random 10 key dimensions
np.random.seed(42)
NTREES = 20
NKEY = 10
key_dims = [np.sort(np.random.choice(50, NKEY, replace=False)) for _ in range(NTREES)]

print(f"  Building {NTREES} trees (each {NKEY}/50 dims)...", flush=True)
t0_all = time.time()
forest = [faiss.IndexFlatL2(NKEY) for _ in range(NTREES)]
for t in range(NTREES):
    kd = key_dims[t]
    forest[t].add(lib_pca50[:, kd])
kdf_build = (time.time()-t0_all)*1000
print(f"  Build: {kdf_build:.0f}ms", flush=True)

# Search all trees → pool candidates → rerank
pred = np.zeros(NQ, dtype=np.float32)
t0_all = time.time()
for qi in range(NQ):
    cand_set = set()
    for t in range(NTREES):
        kd = key_dims[t]
        D, I = forest[t].search(q_pca50[qi:qi+1, kd], 10)
        cand_set.update(I[0].tolist())
    cand = np.array(list(cand_set))
    if len(cand) == 0:
        pred[qi] = 0
    else:
        dots = lib_norm[cand] @ q_norm[qi]
        pred[qi] = lib_thick[cand[dots.argmax()]]
su = (time.time()-t0_all)/NQ*1e6
p1, p5, med = accuracy(pred, q_thick)
accel = bf_time_us/su
r = {'method':'KDF-50D(20trees)','p1nm':p1,'p5nm':p5,'medae':med,
     'build_ms':kdf_build,'search_us':su,'accel':accel}
print(f"  KDF-50D(20trees)                                : P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  MedAE={med:.2f}nm  "
      f"build={kdf_build:.0f}ms  search={su:.0f}us/q  accel={accel:.0f}×", flush=True)
results.append(r)
del forest

# ========================================================================
# Summary
# ========================================================================
print("\n" + "="*80, flush=True)
print(f"{'SUMMARY':^80}", flush=True)
print("="*80, flush=True)
print(f"{'Method':30s} {'P1nm':>7s} {'P5nm':>7s} {'MedAE':>8s} {'Build':>8s} {'Search':>9s} {'Accel':>8s}", flush=True)
print("-"*80, flush=True)
for r in results:
    print(f"{r['method']:30s} {r['p1nm']*100:>6.1f}% {r['p5nm']*100:>6.1f}% "
          f"{r['medae']:>7.2f}nm {r['build_ms']:>7.0f}ms {r['search_us']:>8.0f}us "
          f"{r['accel']:>6.0f}×", flush=True)

with open(f'{OUT}/benchmark_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT}/benchmark_v2.json", flush=True)