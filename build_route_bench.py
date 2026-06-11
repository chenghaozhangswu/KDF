# build_route_bench.py — 601D L2 routing for 4 materials + real data validation
import numpy as np, os, time, json

wl_lib = np.arange(400, 1001, dtype=float)
mats = ['ox','sin','soi','cauthy']
OUT = r'D:\kd_forest_v2\bench_data'

# Step 1: load libraries and sub-sample routing set (every 200)
print("Loading and sub-sampling routing set (1/200)...")
routing_spec_list = []
routing_label_list = []

for mi, mn in enumerate(mats):
    s = np.fromfile(f'{OUT}\\spec_{mn}.bin', dtype=np.float32)
    s = s.reshape(-1, 601).astype(np.float64)
    t = np.fromfile(f'{OUT}\\thick_{mn}.bin', dtype=np.float32)
    
    # Sub-sample every 200
    idx = np.arange(0, len(s), 200)
    sr = s[idx]
    tr = t[idx]
    
    # L2 normalize
    norms = np.linalg.norm(sr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sr = sr / norms
    
    routing_spec_list.append(sr)
    routing_label_list.append(np.full(len(sr), mi, dtype=np.int32))
    print(f"  {mn}: {len(sr)} routing points (from {len(s)})")

routing_spec = np.concatenate(routing_spec_list, axis=0).astype(np.float32)
routing_label = np.concatenate(routing_label_list, axis=0)
print(f"  Total routing set: {routing_spec.shape}")

# Step 2: full library for search (sub-sample for speed, every 50 for each material = 10K per material)
print("\nLoading full search library (sub-sampled every 50)...")
search_spec_list = []
search_thick_list = []
search_label_list = []

for mi, mn in enumerate(mats):
    s = np.fromfile(f'{OUT}\\spec_{mn}.bin', dtype=np.float32)
    s = s.reshape(-1, 601).astype(np.float64)
    t = np.fromfile(f'{OUT}\\thick_{mn}.bin', dtype=np.float32)
    
    # Every 50 for search = 10K per material (enough for benchmark)
    idx = np.arange(0, len(s), 50)
    sr = s[idx]
    tr = t[idx]
    
    norms = np.linalg.norm(sr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sr = sr / norms
    
    search_spec_list.append(sr)
    search_thick_list.append(tr)
    search_label_list.append(np.full(len(sr), mi, dtype=np.int32))

search_spec = np.concatenate(search_spec_list, axis=0).astype(np.float32)
search_thick = np.concatenate(search_thick_list, axis=0)
search_label = np.concatenate(search_label_list, axis=0)
print(f"  Total search set: {search_spec.shape}")

# Step 3: build KDT on routing set
print("\nBuilding routing KDT...")
from scipy.spatial import KDTree as KDTree_scipy
t0 = time.time()
routing_tree = KDTree_scipy(routing_spec, leafsize=16)
print(f"  Built in {time.time()-t0:.1f}s")

# Step 4: test on real data
print("\n=== Real data routing test ===")
real_files = []
for dirname in ['OX','SIN','SOI','CAUTYONGLASS']:
    base = rf'D:\kd_forest_v2\test_data\CE\{dirname}'
    gt = dirname[:4].lower()
    if gt == 'caut': gt = 'cauthy'
    for fn in sorted(os.listdir(base)):
        if not fn.endswith('.csv'): continue
        d = np.loadtxt(os.path.join(base, fn), delimiter=',', skiprows=2)
        I = np.interp(wl_lib, d[:,0], d[:,1])
        I = I / np.linalg.norm(I)
        real_files.append((os.path.basename(fn), gt, I))

import re
route_correct = 0
search_correct = 0

print(f"\n{'File':30s} {'GT':8s} {'Route':8s} {'->Search':10s} {'Dist':10s} {'Thick':>8s} {'OK':>4s}")
print('-'*80)

for fname, gt, I in real_files:
    # Route query
    d_route, idx_route = routing_tree.query(I, k=1)
    route_label = routing_label[idx_route]
    route_mn = mats[route_label]
    
    route_ok = 'OK' if route_mn == gt else 'XX'
    if route_ok == 'OK': route_correct += 1
    
    # Now search within predicted material
    mi = route_label
    search_idx = np.where(search_label == mi)[0]
    if len(search_idx) > 0:
        search_subset = search_spec[search_idx]
        diffs = search_subset - I.astype(np.float32).reshape(1, -1)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        best_search_idx = search_idx[dists.argmin()]
        best_thick = search_thick[best_search_idx]
        best_dist = dists.min()
        search_label_pred = mats[search_label[best_search_idx]]
    else:
        best_thick = -1
        best_dist = -1
        search_label_pred = 'N/A'
    
    search_ok = 'OK' if search_label_pred == gt else 'XX'
    if search_ok == 'OK': search_correct += 1
    
    print(f"{fname:30s} {gt:8s} {route_mn:8s} "
          f"{search_label_pred:10s} {best_dist:>10.6f} {best_thick:>8.1f} "
          f"r:{route_ok} s:{search_ok}")

print(f"\n{'='*50}")
print(f"Routing accuracy: {route_correct}/{len(real_files)} = {100*route_correct/len(real_files):.1f}%")
print(f"Search accuracy:  {search_correct}/{len(real_files)} = {100*search_correct/len(real_files):.1f}%")

# Step 5: also benchmark on simulation queries (2000 from each material)
print("\n\n=== Simulation data routing test ===")
query_spec_list = []
query_thick_list = []
query_label_list = []

for mi, mn in enumerate(mats):
    s = np.fromfile(f'{OUT}\\spec_{mn}.bin', dtype=np.float32)
    s = s.reshape(-1, 601).astype(np.float64)
    t = np.fromfile(f'{OUT}\\thick_{mn}.bin', dtype=np.float32)
    
    # Take one per 250 = 2000 queries per material
    idx = np.arange(0, len(s), 250)
    sr = s[idx]
    norms = np.linalg.norm(sr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sr = sr / norms
    
    query_spec_list.append(sr)
    query_thick_list.append(t[idx])
    query_label_list.append(np.full(len(sr), mi, dtype=np.int32))

query_spec = np.concatenate(query_spec_list, axis=0).astype(np.float32)
query_thick = np.concatenate(query_thick_list, axis=0)
query_label = np.concatenate(query_label_list, axis=0)
print(f"  Total queries: {query_spec.shape}")

t0 = time.time()
q_correct = 0
thick_errors = []
for i in range(len(query_spec)):
    I = query_spec[i]
    # Route
    d_route, idx_route = routing_tree.query(I, k=1)
    route_label = routing_label[idx_route]
    
    # Search within predicted material
    mi = route_label
    search_idx = np.where(search_label == mi)[0]
    search_subset = search_spec[search_idx]
    # L2 in 601D
    diff = search_subset.astype(np.float64) - I.astype(np.float64).reshape(1, -1)
    dists = np.sqrt(np.sum(diff**2, axis=1))
    best_search_idx = search_idx[dists.argmin()]
    
    gt_mn = query_label[i]
    pred_mn = search_label[best_search_idx]
    
    gt_t = query_thick[i]
    pred_t = search_thick[best_search_idx]
    
    if pred_mn == gt_mn:
        q_correct += 1
        thick_errors.append(abs(pred_t - gt_t))

t_elapsed = time.time() - t0
print(f"  Accuracy: {q_correct}/{len(query_spec)} = {100*q_correct/len(query_spec):.1f}%")
print(f"  Time: {t_elapsed:.2f}s for {len(query_spec)} queries = {t_elapsed/len(query_spec)*1e6:.1f} μs/q")
print(f"  Thick errors: P1nm={np.mean(np.array(thick_errors)<=1)*100:.1f}% "
      f"MedAE={np.median(thick_errors):.2f}nm")

# Save routing tree data for C++ port
print("\n\nSaving routing data for C++...")
routing_spec.tofile(f'{OUT}\\route_601d_4m.bin')
routing_label.astype(np.int32).tofile(f'{OUT}\\route_label_4m.bin')
# Also save search set
np.savez(f'{OUT}\\search_4m.npz',
         spec=search_spec, thick=search_thick, label=search_label)
print("  Saved!")
