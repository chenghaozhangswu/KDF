"""
bench_bf_balanced_v2.py — 完全类平衡 BF 路由
每材料各 1000 点（stride=500），总计 4000 点。
"""

import numpy as np, time, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
NW = 601
N_PER_CLASS = 2000  # 再多点

print("=== 完全类平衡 BF（每材料 2000 点）===\n")

rspecs = {}
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = max(1, n // N_PER_CLASS)
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0] = 1; sel /= nrms
    rspecs[m] = sel.astype(np.float32)
    print(f"  {m}: {sel.shape[0]} 点（步长{stride}）")
    del fp

# 组合（用于全局 BF 对比）
rspec_all = np.vstack([rspecs[m] for m in MATS]).astype(np.float32)
rlab_all = np.concatenate([[mi]*len(rspecs[m]) for mi, m in enumerate(MATS)]).astype(np.int32)
print(f"\n  总计: {len(rspec_all)} 点, {rspec_all.nbytes/1024:.1f} KB")

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

# 全局 BF
n_pass = 30
t0 = time.perf_counter()
pm_c = [0]*4
for _ in range(n_pass):
    scores = qspecs @ rspec_all.T
    preds = rlab_all[np.argmax(scores, axis=1)]
    for mi in range(4):
        pm_c[mi] += np.sum((qlabs == mi) & (preds == mi))
dt = (time.perf_counter() - t0) * 1e6 / (n_pass * len(qspecs))

for mi, m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]//n_pass:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c[mi]/(n_pass*np.sum(qlabs==mi)):.1f}%")
acc = 100*sum(pm_c)//n_pass / len(qspecs)
print(f"  {'总':>8}: {sum(pm_c)//n_pass:>4d}/{len(qspecs):>4d} = {acc:.1f}%")
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
for fn, gt, spec in samples:
    scores = spec[None,:] @ rspec_all.T
    best_i = np.argmax(scores[0])
    best_m = MATS[rlab_all[best_i]]
    ok = (best_m == gt)
    if ok: r_correct += 1
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {'OK' if ok else 'XX'}")
print(f"\n  结果: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")

# 如果不达标，看 SOI 间距
print("\n--- 分析 SOI 覆盖 ---")
soi_pts = rspecs["soi"]
soi_cnt = len(soi_pts)
# 对真实 SOI 样品的最近距离
soi_real = [(fn,spec) for fn,gt,spec in samples if gt=='soi']
for fn, spec in soi_real:
    scores = spec[None,:] @ soi_pts.T
    best_d = 2 - 2*scores.max()
    best_other = 999.0
    for m in [m for m in MATS if m != 'soi']:
        s2 = spec[None,:] @ rspecs[m].T
        d2 = 2 - 2*s2.max()
        if d2 < best_other: best_other = d2
    print(f"  {fn:>28}: 最近 soi={best_d:.4f}, 最近别类={best_other:.4f}, 差距={best_other-best_d:.4f}")

print(f"\n===== DONE =====")
