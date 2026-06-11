"""
bench_bf_balanced.py — 类平衡路由：每材料独立 BF，取最小距离
OX: 800, SIN: 800, SOI: 10000（密采样）, CAUTHY: 400
查询在每个材料里独立找最近邻，取最小距离的类 → 天然抗不平衡
"""

import numpy as np, time, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
STRIDE = {"ox": 625, "sin": 625, "soi": 50, "cauthy": 1250}
NW = 601

print("=== 类平衡 BF 路由（独立最近邻，最小距离分类）===\n")

rspecs = {}  # mat -> (N×601) L2 normed
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = STRIDE[m]
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0] = 1; sel /= nrms
    rspecs[m] = sel.astype(np.float32)
    print(f"  {m}: {sel.shape[0]} 点（步长{stride}）")
    del fp

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
    qspecs.append(sel.astype(np.float32)); qlabs.extend([mi]*len(sel))
    del fp
qspecs = np.vstack(qspecs)
qlabs = np.array(qlabs, dtype=np.int32)

# 每材料独立 BF，取最小 L2 距离
# scores[i][m] = max(qspecs[i] · rspecs[m])
# 等价于 min distance = 2 - 2*max_score
n_pass = 30
t0 = time.perf_counter()
pm_c = [0]*4
for _ in range(n_pass):
    best_D = np.full(len(qspecs), np.inf)
    best_M = np.full(len(qspecs), -1, dtype=np.int32)
    for mi, m in enumerate(MATS):
        # 内积 → 最大内积最近
        scores = qspecs @ rspecs[m].T
        max_s = scores.max(axis=1)
        d = 2 - 2 * max_s  # L2 distance from dot product
        mask = d < best_D
        best_D[mask] = d[mask]
        best_M[mask] = mi
    for mi in range(4):
        pm_c[mi] += np.sum((qlabs == mi) & (best_M == mi))
dt = (time.perf_counter() - t0) * 1e6 / (n_pass * len(qspecs))

for mi, m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]//n_pass:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c[mi]/(n_pass*np.sum(qlabs==mi)):.1f}%")
print(f"  {'总':>8}: {sum(pm_c)//n_pass:>4d}/{len(qspecs):>4d} = {100*sum(pm_c)/(n_pass*len(qspecs)):.1f}%")
print(f"  延迟: {dt:.3f} μs/q")

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
        samples.append((os.path.basename(fn), gt, spec.astype(np.float32)))

r_correct = 0
t0r = time.perf_counter()
for fn, gt, spec in samples:
    best_d = np.inf
    best_m = ''
    for mi, m in enumerate(MATS):
        scores = spec[None,:] @ rspecs[m].T
        max_s = scores.max()
        d = 2 - 2 * max_s
        if d < best_d:
            best_d = d
            best_m = m
    ok = (best_m == gt)
    if ok: r_correct += 1
    d_str = ' '.join(f"{m}:{2-2*(spec[None,:]@rspecs[m].T).max():.4f}" for m in MATS)
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {best_d**0.5:.4f}  {'OK' if ok else 'XX'}")
dt_r = (time.perf_counter() - t0r) * 1e6 / len(samples)
print(f"\n  结果: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")
print(f"  延迟: {dt_r:.0f} μs/q")
print(f"\n===== DONE =====")
