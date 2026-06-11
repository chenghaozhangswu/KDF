"""
bench_tpl.py — 最简模板匹配：4 材料均值光谱分类
不建树、不子采样、不降维。就 4 个均值向量，L2 最近即分类。
"""

import numpy as np, time, os, glob

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
NW = 601

print("=== 模板匹配（4 材料均值）=== \n")

# --- 计算均值光谱 (memmap, 累积求和, 不解压) ---
means = {}
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    # 累积求和
    mean = fp.mean(axis=0)
    nrm = np.linalg.norm(mean)
    if nrm: mean /= nrm
    means[m] = mean
    print(f"  {m}: {n} 条, 均值 L2归一化")
    del fp

print(f"\n  均值向量大小: 4 × {NW} × 4bytes = {4*NW*4/1024:.1f} KB")

# --- 仿真查询 ---
print("\n加载仿真查询集...")
qspecs, qlabs = [], []
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = max(1, n // 1502)
    sel = fp[::stride].copy()
    # L2
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0]=1; sel /= nrms
    qspecs.append(sel); qlabs.extend([mi]*len(sel))
    del fp
qspecs = np.vstack(qspecs)
qlabs = np.array(qlabs)
print(f"  仿真查询: {len(qspecs)} 条, {qspecs.nbytes/1e6:.1f} MB")

# 均值矩阵
M = np.vstack([means[m] for m in MATS])  # 4×601

print("\n--- 仿真查询 ---")
t0 = time.perf_counter()
n_pass = 500
pm_c = [0]*4
for _ in range(n_pass):
    # 4×601 × 601×N = 4×N 距离矩阵
    # 展开: query=qspecs, 对每个q算 dists = ||q - M[i]||^2 = ||q||^2 + ||M[i]||^2 - 2q·M[i]
    # 但 q 已 L2 归一化, M 也 L2 归一化, 所以 dists = 2 - 2q·M[i] 等价于最大内积
    scores = qspecs @ M.T  # N×4
    preds = np.argmax(scores, axis=1)
    for mi in range(4):
        pm_c[mi] += np.sum((qlabs == mi) & (preds == mi))
dt = (time.perf_counter() - t0) * 1e6 / (n_pass * len(qspecs))

for mi, m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]//n_pass:>5d}/{np.sum(qlabs==mi):>5d} = {100*pm_c[mi]/(n_pass*np.sum(qlabs==mi)):.1f}%")
tot_c = sum(pm_c)//n_pass
print(f"  {'总':>8}: {tot_c:>5d}/{len(qspecs):>5d} = {100*tot_c/len(qspecs):.1f}%")
print(f"  延迟: {dt:.3f} μs/q（矩阵乘法）")

# --- 真实 CSV ---
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

print(f"  {len(samples)} 条\n")
print(f"  {'文件':>28}  {'GT':>6}  {'→':>6}  {'Dist':>7}  {'OK'}")
print(f"  {'-'*55}")
r_correct = 0
t0r = time.perf_counter()
for fn, gt, spec in samples:
    best_d = float('inf')
    best_m = ''
    all_d = {}
    for mj_name in MATS:
        d = np.sum((spec - means[mj_name])**2)
        all_d[mj_name] = d**0.5
        if d < best_d:
            best_d = d
            best_m = mj_name
    ok = (best_m == gt)
    if ok: r_correct += 1
    dist_str = ' '.join(f"{m}={all_d[m]:.3f}" for m in MATS)
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {best_d**0.5:.4f}  {'OK' if ok else 'XX'}")
    print(f"  {'':>28}  4D: {dist_str}")
dt_r = (time.perf_counter() - t0r) * 1e6 / len(samples)
print(f"\n  结果: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")
print(f"  延迟: {dt_r:.2f} μs/q")
print("\n===== DONE =====")
