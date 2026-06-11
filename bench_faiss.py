"""
KD-Forest v2 benchmark: FAISS HNSW + BF-601D + cKDTree + KDF
Uses correct TMM library from D:/kd_forest_v2/bench_data/
"""
import numpy as np, time, json, faiss, os, sys
from scipy.spatial import cKDTree

DATA = r'D:\kd_forest_v2\bench_data'
OUT = r'D:\kd_forest_v2\results'
os.makedirs(OUT, exist_ok=True)

# === Load data ===
print("Loading data...", flush=True)
lib_pca50 = np.fromfile(f'{DATA}/lib_pca50.bin', dtype=np.float32).reshape(-1, 50)
lib_thick = np.fromfile(f'{DATA}/lib_thick.bin', dtype=np.float32)
lib_mat = np.repeat([0,1,2], 500000)  # 3 materials × 500k

q_pca50 = np.fromfile(f'{DATA}/query_pca50.bin', dtype=np.float32).reshape(-1, 50)
q_thick = np.fromfile(f'{DATA}/query_thick.bin', dtype=np.float32)
q_mat = np.fromfile(f'{DATA}/query_mat.bin', dtype=np.int32)

N, D50 = lib_pca50.shape  # 1.5M, 50
NQ = len(q_thick)  # 1500
NLIB = N // 3  # 500k per material
print(f"Library: {N} points ({NLIB} per mat × 3), Queries: {NQ}, 50D PCA", flush=True)

# Load 601D for reranking / BF
print("Loading 601D spec data...", flush=True)
t0 = time.time()
lib_spec = np.fromfile(f'{DATA}/lib_all_601d.bin', dtype=np.float32).reshape(-1, 601)
q_spec = np.fromfile(f'{DATA}/query_spec.bin', dtype=np.float32).reshape(-1, 601)
print(f"  Loaded: {time.time()-t0:.1f}s", flush=True)

# Normalize 601D for cosine similarity
print("Normalizing 601D...", flush=True)
t0 = time.time()
lib_norm = lib_spec.copy()
lib_norm /= np.linalg.norm(lib_norm, axis=1, keepdims=True)
q_norm = q_spec / np.linalg.norm(q_spec, axis=1, keepdims=True)
del lib_spec, q_spec  # save memory
print(f"  Normalized: {time.time()-t0:.1f}s", flush=True)

def rerank(candidates, q_norm, q_start, q_end):
    """Rerank candidates with 601D cosine, return best per query"""
    pred = np.zeros(q_end - q_start, dtype=np.float32)
    for i, qi in enumerate(range(q_start, q_end)):
        cand = candidates[qi - q_start]
        dots = lib_norm[cand] @ q_norm[qi]
        best = cand[dots.argmax()]
        pred[i] = lib_thick[best]
    return pred

def accuracy(pred, gt):
    p1 = np.mean(np.abs(pred - gt) <= 1.0)
    p5 = np.mean(np.abs(pred - gt) <= 5.0)
    med = np.median(np.abs(pred - gt))
    return p1, p5, med

def fmt(r):
    return f"P1nm={r['p1nm']*100:.1f}%  P5nm={r['p5nm']*100:.1f}%  MedAE={r['medae']:.2f}nm  {r.get('build_ms', 'N/A'):>8}ms build  {r.get('search_us', 0):>8.0f}us/q  accel={r.get('accel', 0):>6.0f}×"

results = []
# Use BF time as baseline for acceleration
bf_time_us = None

# ============================================================
# 1. BF-601D (FAISS IndexFlatIP)
# ============================================================
print("\n" + "="*60, flush=True)
print("1. BF-601D (FAISS IndexFlatIP, full 1.5M)", flush=True)
print("="*60, flush=True)

t0 = time.time()
bf_index = faiss.IndexFlatIP(601)
bf_index.add(lib_norm)
build_ms = (time.time() - t0) * 1000
print(f"  Index build: {build_ms:.0f}ms", flush=True)

t0 = time.time()
D, I = bf_index.search(q_norm, 1)
bf_search_s = (time.time() - t0)
bf_time_us = bf_search_s / NQ * 1e6
bf_pred = lib_thick[I[:, 0]]
p1, p5, med = accuracy(bf_pred, q_thick)
bf_r = {'method': 'BF-601D', 'p1nm': float(p1), 'p5nm': float(p5),
        'medae': float(med), 'build_ms': float(build_ms),
        'search_us': float(bf_time_us), 'accel': 1.0}
print(f"  {fmt(bf_r)}", flush=True)
results.append(bf_r)

# Clean BF to free memory
del bf_index

# ============================================================
# 2. KDT-50D + cosine rerank (exact, via FAISS)
# ============================================================
print("\n" + "="*60, flush=True)
print("2. KDT-50D + rerank (via FAISS IndexFlatIP on 50D)", flush=True)
print("="*60, flush=True)

# Use FAISS FlatL2 on 50D (L2 = same metric as C++ KD-Tree)
K_RERANK = 50
t0 = time.time()
kdt_index = faiss.IndexFlatL2(50)
kdt_index.add(lib_pca50)
kdt_build = (time.time() - t0) * 1000
print(f"  Index build: {kdt_build:.0f}ms", flush=True)

t0 = time.time()
D50, I50 = kdt_index.search(q_pca50, K_RERANK)  # get top 50 by 50D L2
kdt_search_s = (time.time() - t0)
kdt_time_us = kdt_search_s / NQ * 1e6

# Rerank
print("  Reranking with 601D cosine...", flush=True)
t0 = time.time()
# For each query, rerank the K=50 candidates with full 601D cosine
kdt_pred = np.zeros(NQ, dtype=np.float32)
for qi in range(NQ):
    cand = I50[qi]
    dots = lib_norm[cand] @ q_norm[qi]
    best = cand[dots.argmax()]
    kdt_pred[qi] = lib_thick[best]
kdt_rerank_s = time.time() - t0

p1, p5, med = accuracy(kdt_pred, q_thick)
kdt_r = {'method': 'KDT-50D+Rerank', 'p1nm': float(p1), 'p5nm': float(p5),
         'medae': float(med), 'build_ms': float(kdt_build),
         'search_us': float(kdt_time_us),
         'rerank_us': float(kdt_rerank_s / NQ * 1e6),
         'accel': float(bf_time_us / (kdt_search_s / NQ * 1e6)) if bf_time_us else 0}
print(f"  {fmt(kdt_r)}", flush=True)
results.append(kdt_r)

del kdt_index, D50, I50

# ============================================================
# 3. FAISS HNSW on 50D PCA (user's main request! ± rerank)
# ============================================================
print("\n" + "="*60, flush=True)
print("3. FAISS HNSW on 50D PCA + cosine rerank", flush=True)
print("="*60, flush=True)

# Sweep HNSW parameters
hnsw_configs = [
    # (M, efConstruction, efSearch, label)
    (16, 200, 32,   "HNSW-50D(M16,efS=32)"),
    (16, 200, 64,   "HNSW-50D(M16,efS=64)"),
    (16, 200, 128,  "HNSW-50D(M16,efS=128)"),
    (16, 200, 256,  "HNSW-50D(M16,efS=256)"),
    (16, 200, 512,  "HNSW-50D(M16,efS=512)"),
    (8,  200, 64,   "HNSW-50D(M8,efS=64)"),
    (24, 200, 64,   "HNSW-50D(M24,efS=64)"),
]

for M, efC, efS, label in hnsw_configs:
    print(f"\n  --- {label} ---", flush=True)
    t0 = time.time()
    hnsw = faiss.IndexHNSWFlat(50, M)
    hnsw.hnsw.efConstruction = efC
    hnsw.add(lib_pca50)
    hs_build = (time.time() - t0) * 1000
    print(f"    Build: {hs_build:.0f}ms", flush=True)
    
    hnsw.hnsw.efSearch = efS
    t0 = time.time()
    D_hnsw, I_hnsw = hnsw.search(q_pca50, K_RERANK)
    hs_search = (time.time() - t0) / NQ * 1e6
    print(f"    Search: {hs_search:.0f}us/q", flush=True)
    
    # Rerank with 601D cosine
    t0 = time.time()
    hs_pred = np.zeros(NQ, dtype=np.float32)
    for qi in range(NQ):
        cand = I_hnsw[qi]
        dots = lib_norm[cand] @ q_norm[qi]
        best = cand[dots.argmax()]
        hs_pred[qi] = lib_thick[best]
    hs_rerank = (time.time() - t0) / NQ * 1e6
    
    p1, p5, med = accuracy(hs_pred, q_thick)
    accel = bf_time_us / (hs_search + hs_rerank) if bf_time_us else 0
    hs_r = {'method': label, 'p1nm': float(p1), 'p5nm': float(p5),
            'medae': float(med), 'build_ms': float(hs_build),
            'search_us': float(hs_search), 'rerank_us': float(hs_rerank),
            'accel': float(accel)}
    print(f"    {fmt(hs_r)}", flush=True)
    results.append(hs_r)
    del hnsw, D_hnsw, I_hnsw

# ============================================================
# 4. FAISS IndexFlatIP on 50D (exact 50D = best possible KDT-50D)
# ============================================================
print("\n" + "="*60, flush=True)
print("4. BF-50D (exact, FAISS IndexFlatIP)", flush=True)
print("="*60, flush=True)

t0 = time.time()
if50 = faiss.IndexFlatL2(50)
if50.add(lib_pca50)
if_build = (time.time() - t0) * 1000

t0 = time.time()
D_if50, I_if50 = if50.search(q_pca50, K_RERANK)
if_search = (time.time() - t0) / NQ * 1e6

t0 = time.time()
if_pred = np.zeros(NQ, dtype=np.float32)
for qi in range(NQ):
    cand = I_if50[qi]
    dots = lib_norm[cand] @ q_norm[qi]
    best = cand[dots.argmax()]
    if_pred[qi] = lib_thick[best]
if_rerank = (time.time() - t0) / NQ * 1e6

p1, p5, med = accuracy(if_pred, q_thick)
accel = bf_time_us / (if_search + if_rerank) if bf_time_us else 0
if_r = {'method': f'BF-50D+Rerank(K={K_RERANK})', 'p1nm': float(p1), 'p5nm': float(p5),
        'medae': float(med), 'build_ms': float(if_build),
        'search_us': float(if_search), 'rerank_us': float(if_rerank),
        'accel': float(accel)}
print(f"  {fmt(if_r)}", flush=True)
results.append(if_r)
del if50, D_if50, I_if50

# ============================================================
# Summary table
# ============================================================
print("\n" + "="*60, flush=True)
print("SUMMARY TABLE", flush=True)
print("="*60, flush=True)
print(f"{'Method':40s} {'P1nm':>7s} {'P5nm':>7s} {'MedAE':>7s} {'Build':>8s} {'Search':>8s} {'Accel':>8s}", flush=True)
print("-"*85, flush=True)
for r in results:
    print(f"{r['method']:40s} {r['p1nm']*100:>6.1f}% {r['p5nm']*100:>6.1f}% "
          f"{r['medae']:>6.2f}nm {r['build_ms']:>7.0f}ms {r['search_us']:>7.0f}us "
          f"{r['accel']:>6.0f}×", flush=True)

# Save
with open(f'{OUT}/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}/benchmark_results.json", flush=True)
