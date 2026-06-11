"""
bench_bf_soi_2d.py — SOI 用 2D 网格采样，其他用 1D stride
SOI: 500 × 1000 × 601 网格，取 top 步长 5、BOX 步长 10 → 100×100 = 10K 点
"""

import numpy as np, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
NW = 601
STRIDE1D = {"ox": 625, "sin": 625, "cauthy": 1250}

print("=== SOI 2D 网格采样 BF 路由 ===\n")

rspecs = {}
for m in MATS:
    if m == 'soi':
        p = os.path.join(DATA, "spec_soi.bin")
        fp = np.memmap(p, dtype=np.float32, mode='r')
        fp_2d = fp.reshape(500, 1000, NW)  # top=500, box=1000
        sr, sc = 5, 10  # top 每5, BOX 每10 → 100×100=10K
        sel = fp_2d[::sr, ::sc, :].copy().reshape(-1, NW)
        print(f"  soi: 2D sr={sr} sc={sc} -> {len(sel)} pts")
        del fp
    else:
        p = os.path.join(DATA, f"spec_{m}.bin")
        fp = np.memmap(p, dtype=np.float32, mode='r')
        n = len(fp) // NW
        fp = fp.reshape(n, NW)
        stride = STRIDE1D[m]
        sel = fp[::stride].copy()
        print(f"  {m}: stride={stride} -> {len(sel)} pts")
        del fp

    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms == 0] = 1
    sel /= nrms
    rspecs[m] = sel.astype(np.float32)

rspec_all = np.vstack([rspecs[m] for m in MATS])
rlab_all = np.concatenate([[mi] * len(rspecs[m]) for mi, m in enumerate(MATS)])
print(f"  total: {len(rspec_all)} pts, {rspec_all.nbytes/1024:.0f} KB")

# 仿真查询
print("\n--- Simulation ---")
qspecs, qlabs = [], []
for mi, m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp) // NW
    fp = fp.reshape(n, NW)
    stride = max(1, n // 1502)
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms == 0] = 1
    sel /= nrms
    qspecs.append(sel.astype(np.float32))
    qlabs.extend([mi] * len(sel))
    del fp
qspecs = np.vstack(qspecs)
qlabs = np.array(qlabs, dtype=np.int32)

scores = qspecs @ rspec_all.T
preds = rlab_all[np.argmax(scores, axis=1)]
pm_c = [np.sum((qlabs == mi) & (preds == mi)) for mi in range(4)]
for mi, m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c[mi]/np.sum(qlabs==mi):.1f}%")
print(f"  {'total':>8}: {sum(pm_c):>4d}/{len(qspecs):>4d} = {100*sum(pm_c)/len(qspecs):.1f}%")

# 真实 CSV
print("\n--- Real CSV ---")
wl = np.linspace(400, 1000, NW)
dirs = [("OX","ox"),("SIN","sin"),("SOI","soi"),("CAUTYONGLASS","cauthy")]
base = R"D:\kd_forest_v2\test_data\CE"
samples = []
for d, gt in dirs:
    for fn in sorted(glob.glob(os.path.join(base, d, "*.csv"))):
        data = np.loadtxt(fn, delimiter=',', skiprows=2)
        spec = np.interp(wl, data[:, 0], data[:, 1])
        nrm = np.linalg.norm(spec)
        if nrm:
            spec /= nrm
        samples.append((os.path.basename(fn), gt, spec.astype(np.float32)))

r_correct = 0
for fn, gt, spec in samples:
    scores = spec[None, :] @ rspec_all.T
    best_m = MATS[rlab_all[np.argmax(scores[0])]]
    ok = best_m == gt
    r_correct += ok
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {'OK' if ok else 'XX'}")
print(f"\n  result: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")

# 如果不够 99%，试试更密的版本
sim_acc = 100 * sum(pm_c) / len(qspecs)
real_acc = 100 * r_correct / len(samples)
if sim_acc < 99 or real_acc < 100:
    print(f"\n--- retry: denser (sim={sim_acc:.1f}% real={real_acc:.1f}%) ---")
    rspecs2 = {}
    for m in MATS:
        if m == 'soi':
            p = os.path.join(DATA, "spec_soi.bin")
            fp = np.memmap(p, dtype=np.float32, mode='r')
            fp_2d = fp.reshape(500, 1000, NW)
            sr, sc = 3, 5  # denser: 167×200=33,400
            sel = fp_2d[::sr, ::sc, :].copy().reshape(-1, NW)
            print(f"  soi: 2D sr={sr} sc={sc} -> {len(sel)} pts")
            del fp
        else:
            p = os.path.join(DATA, f"spec_{m}.bin")
            fp = np.memmap(p, dtype=np.float32, mode='r')
            n = len(fp)//NW; fp = fp.reshape(n, NW)
            stride = 100  # 5000 pts
            sel = fp[::stride].copy()
            print(f"  {m}: stride={stride} -> {len(sel)} pts")
            del fp
        nrms = np.linalg.norm(sel, axis=1, keepdims=True)
        nrms[nrms==0]=1; sel/=nrms
        rspecs2[m] = sel.astype(np.float32)

    rspec_all2 = np.vstack([rspecs2[m] for m in MATS])
    rlab_all2 = np.concatenate([[mi]*len(rspecs2[m]) for mi,m in enumerate(MATS)])
    print(f"  total: {len(rspec_all2)} pts, {rspec_all2.nbytes/1024:.0f} KB")

    scores2 = qspecs @ rspec_all2.T
    preds2 = rlab_all2[np.argmax(scores2, axis=1)]
    pm_c2 = [np.sum((qlabs==mi)&(preds2==mi)) for mi in range(4)]
    for mi,m in enumerate(MATS):
        print(f"  {m:>8}: {pm_c2[mi]:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c2[mi]/np.sum(qlabs==mi):.1f}%")
    print(f"  {'total':>8}: {sum(pm_c2):>4d}/{len(qspecs):>4d} = {100*sum(pm_c2)/len(qspecs):.1f}%")

    r_correct2 = 0
    for fn,gt,spec in samples:
        scores = spec[None,:] @ rspec_all2.T
        best_m = MATS[rlab_all2[np.argmax(scores[0])]]
        r_correct2 += (best_m==gt)
    print(f"  real: {r_correct2}/{len(samples)} = {100*r_correct2/len(samples):.1f}%")

print("\n===== DONE =====")
