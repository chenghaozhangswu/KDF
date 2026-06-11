"""
bench_bf_99.py — 冲刺 99%
"""
import numpy as np, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
NW = 601

# 方案：SOI 2D sr=2, sc=5 (250×200=50K), 其他 5000
print("=== BF 冲刺 99% ===\n")
rspecs = {}
for m in MATS:
    if m == 'soi':
        fp = np.memmap(os.path.join(DATA,"spec_soi.bin"), dtype=np.float32, mode='r')
        g = fp.reshape(500, 1000, NW)
        sel = g[::2, ::5, :].copy().reshape(-1, NW)  # 250×200=50K
        print(f"  soi: sr=2 sc=5 -> {len(sel)} pts")
        del fp
    else:
        fp = np.memmap(os.path.join(DATA,f"spec_{m}.bin"), dtype=np.float32, mode='r')
        n = len(fp)//NW; fp = fp.reshape(n, NW)
        sel = fp[::100].copy()  # 5000 pts each
        print(f"  {m}: stride=100 -> {len(sel)} pts")
        del fp
    nrms = np.linalg.norm(sel, axis=1, keepdims=True)
    nrms[nrms==0]=1; sel/=nrms
    rspecs[m] = sel.astype(np.float32)

rspec_all = np.vstack([rspecs[m] for m in MATS])
rlab_all = np.concatenate([[mi]*len(rspecs[m]) for mi,m in enumerate(MATS)])
print(f"  total: {len(rspec_all)} pts, {rspec_all.nbytes/1024:.0f} KB")

# 仿真
print("\n--- Simulation ---")
qspecs, qlabs = [], []
for mi,m in enumerate(MATS):
    fp = np.memmap(os.path.join(DATA,f"spec_{m}.bin"), dtype=np.float32, mode='r')
    n = len(fp)//NW; fp = fp.reshape(n, NW)
    stride = max(1, n//1502)
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True); nrms[nrms==0]=1; sel/=nrms
    qspecs.append(sel.astype(np.float32)); qlabs.extend([mi]*len(sel))
    del fp
qspecs = np.vstack(qspecs); qlabs = np.array(qlabs, dtype=np.int32)

scores = qspecs @ rspec_all.T
preds = rlab_all[np.argmax(scores, axis=1)]
pm_c = [np.sum((qlabs==mi)&(preds==mi)) for mi in range(4)]
for mi,m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c[mi]/np.sum(qlabs==mi):.1f}%")
acc = 100*sum(pm_c)/len(qspecs)
print(f"  {'total':>8}: {sum(pm_c):>4d}/{len(qspecs):>4d} = {acc:.1f}%")

# 实测
print("\n--- Real CSV ---")
wl = np.linspace(400,1000,NW)
dirs = [("OX","ox"),("SIN","sin"),("SOI","soi"),("CAUTYONGLASS","cauthy")]
base = R"D:\kd_forest_v2\test_data\CE"
samples = []
for d,gt in dirs:
    for fn in sorted(glob.glob(os.path.join(base,d,"*.csv"))):
        data = np.loadtxt(fn,delimiter=',',skiprows=2)
        spec = np.interp(wl,data[:,0],data[:,1])
        nrm = np.linalg.norm(spec); spec = spec/nrm if nrm else spec
        samples.append((os.path.basename(fn),gt,spec.astype(np.float32)))

rc = 0
for fn,gt,spec in samples:
    scores = spec[None,:] @ rspec_all.T
    best_m = MATS[rlab_all[np.argmax(scores[0])]]
    rc += (best_m==gt)
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {'OK' if best_m==gt else 'XX'}")
print(f"\n  result: {rc}/{len(samples)} = {100*rc/len(samples):.1f}%")

# 如果还不够，显示 SOI 遗漏的厚度分布
if acc < 99.0:
    print(f"\n--- SOI misclassifications (acc={acc:.1f}%) ---")
    # 找出 SOI 误分类的索引
    soi_mask = qlabs == 2  # SOI is index 2
    soi_wrong = (preds != 2) & soi_mask
    soi_wrong_idx = np.where(soi_wrong)[0]
    # 从 SOI 查询序列看被错分到哪个类
    wrong_class_counts = np.bincount(preds[soi_wrong], minlength=4)
    for mi,m in enumerate(MATS):
        if wrong_class_counts[mi] > 0:
            print(f"    -> {m}: {wrong_class_counts[mi]}")
    # 看看是否集中在特定厚度范围
    # 查询的步长是 n//1502, 每个 SOI 查询对应原始库的某个厚度
    fp = np.memmap(os.path.join(DATA,"spec_soi.bin"), dtype=np.float32, mode='r')
    n = len(fp)//NW
    stride_q = max(1, n//1502)
    thick_fp = np.memmap(os.path.join(DATA,"thick_soi.bin"), dtype=np.float32, mode='r')
    thick = thick_fp.reshape(n, 2)
    wrong_idx = np.where(soi_wrong)[0] * stride_q  # approximate original idx
    print(f"  wrong SOI sample indices (approx): {wrong_idx[:20]}")
    for idx in wrong_idx[:10]:
        if idx < len(thick):
            print(f"    idx~{idx}: top={thick[idx,0]:.0f}nm, box={thick[idx,1]:.0f}nm")
    # 看看正确分类的厚度
    soi_right = (preds == 2) & soi_mask
    right_idx = np.where(soi_right)[0] * stride_q
    print(f"  correct SOI sample indices (ex: first 10): {right_idx[:10]}")
    for idx in right_idx[:5]:
        if idx < len(thick):
            print(f"    idx~{idx}: top={thick[idx,0]:.0f}nm, box={thick[idx,1]:.0f}nm")
    del fp, thick_fp

print("\n===== DONE =====")
