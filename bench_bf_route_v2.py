"""
bench_bf_route_v2.py — BF 全点集路由 v2
SOI 加大密度至 10K，其他 800-400，总约 12000 点。
"""

import numpy as np, time, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
STRIDE = {"ox": 625, "sin": 625, "soi": 50, "cauthy": 1250}
NW = 601

print("=== BF 全点集路由 v2（SOI 10K 点）===\n")

rspec, rlab = [], []
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = STRIDE[m]
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0] = 1; sel /= nrms
    rspec.append(sel); rlab.extend([mi]*len(sel))
    print(f"  {m}: {len(sel)} 点（步长{stride}）")
    del fp
rspec = np.vstack(rspec).astype(np.float32)
rlab = np.array(rlab, dtype=np.int32)
print(f"  总: {len(rspec)} 点, {rspec.nbytes/1024:.1f} KB")

# 仿真查询
print("\n--- 仿真查询 ---")
qspecs, qlabs = [], []
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = max(1, n // 1502)
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0]=1; sel /= nrms
    qspecs.append(sel); qlabs.extend([mi]*len(sel))
    del fp
qspecs = np.vstack(qspecs)
qlabs = np.array(qlabs, dtype=np.int32)

# 多次计时
n_pass = 20
t0 = time.perf_counter()
for _ in range(n_pass):
    scores = qspecs @ rspec.T
    preds = rlab[np.argmax(scores, axis=1)]
dt = (time.perf_counter() - t0) * 1e6 / (n_pass * len(qspecs))

pm_c = [np.sum((qlabs==mi)&(preds==mi)) for mi in range(4)]
for mi, m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c[mi]/np.sum(qlabs==mi):.1f}%")
print(f"  {'总':>8}: {sum(pm_c):>4d}/{len(qspecs):>4d} = {100*sum(pm_c)/len(qspecs):.1f}%")
print(f"  BF 延迟: {dt:.3f} μs/q")

# 真实 CSV
print("\n--- 真实 CSV ---")
wl_lib = np.linspace(400, 1000, NW)
dirs = [("OX","ox"),("SIN","sin"),("SOI","soi"),("CAUTYONGLASS","cauthy")]
base = R"D:\kd_forest_v2\test_data\CE"
samples = []
for d, gt in dirs:
    dp = os.path.join(base, d)
    if not os.path.isdir(dp): continue
    for fn in sorted(glob.glob(os.path.join(dp, "*.csv"))):
        data = np.loadtxt(fn, delimiter=',', skiprows=2)
        raw_wl, raw_I = data[:,0], data[:,1]
        spec = np.interp(wl_lib, raw_wl, raw_I)
        nrm = np.linalg.norm(spec)
        if nrm: spec /= nrm
        samples.append((os.path.basename(fn), gt, spec))

r_correct = 0
t0r = time.perf_counter()
for fn, gt, spec in samples:
    scores = spec[None,:] @ rspec.T
    best_i = np.argmax(scores[0])
    best_m = MATS[rlab[best_i]]
    ok = (best_m == gt); 
    if ok: r_correct += 1
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {'OK' if ok else 'XX'}")
dt_r = (time.perf_counter() - t0r) * 1e6 / len(samples)
print(f"\n  结果: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")
print(f"  延迟: {dt_r:.1f} μs/q")

print(f"\n===== DONE =====")
