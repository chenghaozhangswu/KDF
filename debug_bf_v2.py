"""Quick BF-601D debug — per-material oracle vs full"""
import numpy as np, faiss, time

DATA = r'D:\kd_forest_v2\bench_data'
N = 1500000; NL = 500000

print("Loading...", flush=True)
lib_spec = np.memmap(f'{DATA}/lib_all_601d.bin', dtype=np.float32, mode='r').reshape(-1, 601)
lib_thick = np.fromfile(f'{DATA}/lib_thick.bin', dtype=np.float32)
q_spec = np.fromfile(f'{DATA}/query_spec.bin', dtype=np.float32).reshape(-1, 601)
q_thick = np.fromfile(f'{DATA}/query_thick.bin', dtype=np.float32)
q_mat = np.fromfile(f'{DATA}/query_mat.bin', dtype=np.int32)

# Per-material BF with oracle
print("="*60, flush=True)
for m in range(3):
    print(f"\nMaterial {m}:", flush=True)
    mq = np.where(q_mat == m)[0]
    gt = q_thick[mq]
    
    # Normalize this material's library
    t0 = time.time()
    sub = np.array(lib_spec[m*NL:(m+1)*NL])  # copy 500k×601
    sub_norm = sub / np.linalg.norm(sub, axis=1, keepdims=True)
    name = {0:'SiO2',1:'Si3N4',2:'a-Si'}[m]
    print(f"  {name}: {sub_norm.shape} normalized in {time.time()-t0:.1f}s", flush=True)
    
    idx = faiss.IndexFlatIP(601)
    idx.add(sub_norm)
    
    q_sub = q_spec[mq]
    q_norm = q_sub / np.linalg.norm(q_sub, axis=1, keepdims=True)
    
    t0 = time.time()
    D, I = idx.search(q_norm, 1)
    search_us = (time.time()-t0)/len(mq)*1e6
    
    pred = lib_thick[m*NL + I[:,0]]
    p1 = np.mean(np.abs(pred - gt) <= 1.0)
    p5 = np.mean(np.abs(pred - gt) <= 5.0)
    print(f"  P1nm={p1*100:.1f}%  P5nm={p5*100:.1f}%  {search_us:.0f}us/q", flush=True)
    
    # Show thick range stats
    print(f"  GT thick range: {gt.min():.0f}-{gt.max():.0f}nm  Lib thick range: {lib_thick[m*NL:(m+1)*NL].min():.0f}-{lib_thick[m*NL:(m+1)*NL].max():.0f}nm", flush=True)
    
    # Check a few specific queries
    for j in range(3):
        qi = mq[j]
        best_idx = I[j,0]
        gt_t = q_thick[qi]
        pred_t = lib_thick[m*NL + best_idx]
        print(f"  q{qi}: gt={gt_t:.0f}nm pred={pred_t:.0f}nm cos={D[j,0]:.4f}", flush=True)
    
    del idx, sub_norm, q_norm, q_sub, sub