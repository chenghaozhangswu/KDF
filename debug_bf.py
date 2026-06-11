"""Debug BF-601D accuracy — why only 88.7%?"""
import numpy as np, time, faiss

DATA = r'D:\kd_forest_v2\bench_data'
N = 1500000; NL = 500000

print("Loading...", flush=True)
lib_spec = np.memmap(f'{DATA}/lib_all_601d.bin', dtype=np.float32, mode='r').reshape(-1, 601)
lib_thick = np.fromfile(f'{DATA}/lib_thick.bin', dtype=np.float32)
q_spec = np.fromfile(f'{DATA}/query_spec.bin', dtype=np.float32).reshape(-1, 601)
q_thick = np.fromfile(f'{DATA}/query_thick.bin', dtype=np.float32)
q_mat = np.fromfile(f'{DATA}/query_mat.bin', dtype=np.int32)

# Per-material oracle BF (chunked, no full copy)
print("Per-material BF-601D (oracle)...", flush=True)
for m in range(3):
    sub = lib_spec[m*NL:(m+1)*NL]  # 500k × 601
    sub_norm = sub / np.linalg.norm(sub, axis=1, keepdims=True)
    mq = q_mat == m
    q_sub = q_spec[mq]
    q_sub_norm = q_sub / np.linalg.norm(q_sub, axis=1, keepdims=True)
    
    t0 = time.time()
    idx = faiss.IndexFlatIP(601)
    idx.add(sub_norm)
    D, I = idx.search(q_sub_norm, 1)
    build = (time.time()-t0)*1000
    pred = lib_thick[m*NL + I[:,0]]
    gt = q_thick[mq]
    p1 = np.mean(np.abs(pred - gt) <= 1.0)
    print(f"  Mat {m}: P1nm={p1*100:.1f}%  build={build:.0f}ms  queries={mq.sum()}", flush=True)
    del idx, sub_norm, q_sub_norm

# Now full BF with smaller batch
print("\nFull BF-601D (cross-material, chunked)...", flush=True)
t0 = time.time()
pred = np.zeros(1500, dtype=np.float32)
for qi in range(1500):
    dots = np.zeros(N, dtype=np.float32)
    # Chunk the library
    for chunk_start in range(0, N, 100000):
        chunk_end = min(chunk_start + 100000, N)
        lib_chunk = lib_spec[chunk_start:chunk_end]
        lib_chunk_norm = lib_chunk / np.linalg.norm(lib_chunk, axis=1, keepdims=True)
        dots[chunk_start:chunk_end] = lib_chunk_norm @ q_spec[qi:qi+1].T[:,0]
    best = dots.argmax()
    pred[qi] = lib_thick[best]
    
    if qi % 300 == 0 and qi > 0:
        print(f"  {qi}/1500 queries... {time.time()-t0:.0f}s", flush=True)

p1 = np.mean(np.abs(pred - q_thick) <= 1.0)
print(f"\nFull BF-601D: {p1*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

# Check wrong ones
wrong = np.abs(pred - q_thick) > 1.0
print(f"Wrong: {wrong.sum()}/1500", flush=True)
for qi in np.where(wrong)[0][:10]:
    gt_m = q_mat[qi]
    # Find what library index was chosen
    # (we don't have best_idx directly, reconstruct from lib_thick)
    print(f"  q{qi}: mat{gt_m} thick={q_thick[qi]:.0f}nm pred={pred[qi]:.0f}nm")