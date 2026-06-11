"""
bench_bf_route.py — BF 全点集路由（最简方案，回归 KDT 成功的本质）
每个材料隔步长取出样本，所有点放一起 BF，最近点标签即分类。
无训练、无聚类、无降维。
"""

import numpy as np, time, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
NW = 601
# 各材料的采样密度（考虑到 SOI 是 2D 空间，给它更多点）
STRIDE = {"ox": 1250, "sin": 1250, "soi": 312, "cauthy": 2500}
# → ox: 500K/1250=400, sin:400, soi:500K/312≈1602, cauthy:200, 总计≈2602

print("=== BF 全点集路由（暴力搜索原始 601D）===\n")

# --- 构建路由集 ---
rspec, rlab = [], []
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = STRIDE[m]
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0] = 1
    sel /= nrms
    rspec.append(sel)
    rlab.extend([mi] * len(sel))
    print(f"  {m}: {len(sel)} 个路由点（{n}→步长{stride}）")
    del fp

rspec = np.vstack(rspec).astype(np.float32)
rlab = np.array(rlab, dtype=np.int32)
print(f"  总路由集: {len(rspec)} 点, {rspec.nbytes/1024:.1f} KB")

# 预热
_ = np.zeros((1, len(rspec)))

# --- 仿真查询 ---
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
print(f"  查询: {len(qspecs)} 条, {qspecs.nbytes/1e6:.1f} MB")

# BF 路由（单次）
t0 = time.perf_counter()
scores = qspecs @ rspec.T  # N×R
preds = rlab[np.argmax(scores, axis=1)]
dt = (time.perf_counter() - t0) * 1e6 / len(qspecs)

pm_c = [0]*4
for mi in range(4):
    pm_c[mi] = np.sum((qlabs == mi) & (preds == mi))
    tot = np.sum(qlabs == mi)
    print(f"  {MATS[mi]:>8}: {pm_c[mi]:>4d}/{tot:>4d} = {100*pm_c[mi]/tot:.1f}%")
print(f"  {'总':>8}: {sum(pm_c):>4d}/{len(qspecs):>4d} = {100*sum(pm_c)/len(qspecs):.1f}%")
print(f"  BF 延迟: {dt:.3f} μs/q")

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
r_correct = 0
t0r = time.perf_counter()
for fn, gt, spec in samples:
    scores = spec[None,:] @ rspec.T
    best_i = np.argmax(scores[0])
    best_m = MATS[rlab[best_i]]
    ok = (best_m == gt)
    if ok: r_correct += 1
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {'OK' if ok else 'XX'}")
dt_r = (time.perf_counter() - t0r) * 1e6 / len(samples)
print(f"\n  结果: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")
print(f"  延迟: {dt_r:.1f} μs/q")

# --- 如果还不行，加密度 ---
sim_acc = 100 * sum(pm_c) / len(qspecs)
real_acc = 100 * r_correct / len(samples)
if sim_acc < 99 or real_acc < 99:
    print(f"\n--- 精度不足 (仿真 {sim_acc:.1f}%, 实测 {real_acc:.1f}%)，增加路由密度 ---")
    # 试试翻倍：stride 减半
    STRIDE2 = {"ox": 625, "sin": 625, "soi": 156, "cauthy": 1250}
    print(f"  新步长: {STRIDE2}")
    rspec2, rlab2 = [], []
    for mi, m in enumerate(MATS):
        p = os.path.join(DATA, f"spec_{m}.bin")
        fp = np.memmap(p, dtype=np.float32, mode='r')
        n = len(fp) // NW
        fp = fp.reshape(n, NW)
        stride = STRIDE2[m]
        sel = fp[::stride].copy()
        nrms = np.linalg.norm(sel, axis=1, keepdims=True)
        nrms[nrms==0] = 1
        sel /= nrms
        rspec2.append(sel); rlab2.extend([mi]*len(sel))
        print(f"  {m}: {len(sel)} 点（步长{stride}）")
        del fp
    rspec2 = np.vstack(rspec2).astype(np.float32)
    rlab2 = np.array(rlab2, dtype=np.int32)
    print(f"  总: {len(rspec2)} 点, {rspec2.nbytes/1024:.1f} KB")
    
    _ = np.zeros((1, len(rspec2)))
    scores2 = qspecs @ rspec2.T
    preds2 = rlab2[np.argmax(scores2, axis=1)]
    pm_c2 = [np.sum((qlabs==mi)&(preds2==mi)) for mi in range(4)]
    sim_acc2 = 100*sum(pm_c2)/len(qspecs)
    print(f"\n  仿真精度: {sum(pm_c2):>4d}/{len(qspecs):>4d} = {sim_acc2:.1f}%")
    for mi in range(4):
        print(f"    {MATS[mi]:>8}: {pm_c2[mi]:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c2[mi]/np.sum(qlabs==mi):.1f}%")
    
    r_correct2 = 0
    for fn, gt, spec in samples:
        s2 = spec[None,:] @ rspec2.T
        bmi = np.argmax(s2[0])
        if MATS[rlab2[bmi]] == gt: r_correct2 += 1
    real_acc2 = 100*r_correct2/len(samples)
    print(f"  实测精度: {r_correct2}/{len(samples)} = {real_acc2:.1f}%")

print(f"\n===== DONE =====")
