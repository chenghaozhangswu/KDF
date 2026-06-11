"""
bench_bf_dense.py — 全类充足采样
OX=5000, SIN=5000, SOI=10000 (stride=50), CAUTHY=5000 → 总计 25000
"""
import numpy as np, time, os, glob, warnings
warnings.filterwarnings('ignore')

DATA = R"D:\kd_forest_v2\bench_data"
MATS = ["ox", "sin", "soi", "cauthy"]
STRIDE = {"ox": 100, "sin": 100, "soi": 50, "cauthy": 100}
NW = 601

print("=== BF 充足采样（25K 点）===\n")
rspecs = {}
for m in MATS:
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp)//NW; fp = fp.reshape(n, NW)
    sel = fp[::STRIDE[m]].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True); nrms[nrms==0]=1; sel/=nrms
    rspecs[m] = sel.astype(np.float32)
    print(f"  {m}: {len(sel)} 点")
    del fp

rspec_all = np.vstack([rspecs[m] for m in MATS])
rlab_all = np.concatenate([[mi]*len(rspecs[m]) for mi,m in enumerate(MATS)])
print(f"  总: {len(rspec_all)} 点, {rspec_all.nbytes/1024:.1f} KB")

# 仿真查询
print("\n--- 仿真查询 ---")
qspecs, qlabs = [], []
for mi,m in enumerate(MATS):
    p = os.path.join(DATA, f"spec_{m}.bin")
    fp = np.memmap(p, dtype=np.float32, mode='r')
    n = len(fp)//NW; fp = fp.reshape(n, NW)
    stride = max(1, n//1502)
    sel = fp[::stride].copy()
    nrms = np.linalg.norm(sel, axis=1, keepdims=True); nrms[nrms==0]=1; sel/=nrms
    qspecs.append(sel.astype(np.float32)); qlabs.extend([mi]*len(sel))
    del fp
qspecs = np.vstack(qspecs); qlabs = np.array(qlabs, dtype=np.int32)

n_pass = 10
t0 = time.perf_counter()
pm_c = [0]*4
for _ in range(n_pass):
    scores = qspecs @ rspec_all.T
    preds = rlab_all[np.argmax(scores, axis=1)]
    for mi in range(4): pm_c[mi] += np.sum((qlabs==mi)&(preds==mi))
dt = (time.perf_counter()-t0)*1e6/(n_pass*len(qspecs))

for mi,m in enumerate(MATS):
    print(f"  {m:>8}: {pm_c[mi]//n_pass:>4d}/{np.sum(qlabs==mi):>4d} = {100*pm_c[mi]/(n_pass*np.sum(qlabs==mi)):.1f}%")
print(f"  {'总':>8}: {sum(pm_c)//n_pass:>4d}/{len(qspecs):>4d} = {100*sum(pm_c)//n_pass/len(qspecs)*100:.1f}%")
print(f"  延迟: {dt:.3f} μs/q")

# 真实 CSV
print("\n--- 真实 CSV ---")
wl_lib = np.linspace(400,1000,NW)
dirs = [("OX","ox"),("SIN","sin"),("SOI","soi"),("CAUTYONGLASS","cauthy")]
base = R"D:\kd_forest_v2\test_data\CE"
samples = []
for d,gt in dirs:
    for fn in sorted(glob.glob(os.path.join(base,d,"*.csv"))):
        data = np.loadtxt(fn,delimiter=',',skiprows=2)
        spec = np.interp(wl_lib,data[:,0],data[:,1])
        nrm = np.linalg.norm(spec); spec = spec/nrm if nrm else spec
        samples.append((os.path.basename(fn),gt,spec.astype(np.float32)))

r_correct = 0
for fn,gt,spec in samples:
    scores = spec[None,:] @ rspec_all.T
    best_m = MATS[rlab_all[np.argmax(scores[0])]]
    ok = best_m==gt; r_correct += ok
    print(f"  {fn:>28}  {gt:>6}  {best_m:>6}  {'OK' if ok else 'XX'}")
print(f"\n  结果: {r_correct}/{len(samples)} = {100*r_correct/len(samples):.1f}%")
print(f"\n===== DONE =====")
