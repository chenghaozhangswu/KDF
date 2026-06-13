
import numpy as np, time
from scipy.spatial import KDTree

MPATH = r'D:\kd_forest_v2_gh\src\multi'
MNAMES = ['ox','sin','soi','cauthy']; NMAT=4; N=601; NQ=200

print("="*70)
print("KD-Forest v2 (修复噪声不一致 Bug)")
print("="*70)

print("\n[1] 加载 10K/材料 数据...")
specs, thick, label = [], [], []
for m in range(NMAT):
    fn = f'{MPATH}/lib_{MNAMES[m]}_n_10k.bin'
    ft = f'{MPATH}/lib_{MNAMES[m]}_thick_10k.bin'
    s = np.fromfile(fn, dtype=np.float32).reshape(-1, N)
    t = np.fromfile(ft, dtype=np.float32)
    specs.append(s); thick.append(t)
    label.append(np.full(len(s), m, dtype=np.int32))
specs = np.concatenate(specs, 0); thick = np.concatenate(thick, 0)
label = np.concatenate(label, 0)
print(f'  总库: {len(specs)} 条')

print("\n[2] 加载路由...")
rfeat = np.fromfile(f'{MPATH}/route_feat_norm.bin', dtype=np.float32).reshape(-1,10)
rlab  = np.fromfile(f'{MPATH}/route_labels.bin', dtype=np.int32)
rmean = np.fromfile(f'{MPATH}/route_mean.bin', dtype=np.float32)
rstd  = np.fromfile(f'{MPATH}/route_std.bin', dtype=np.float32)

print("\n[3] 加载 PCA...")
pca_data = {}
for m in range(NMAT):
    pca_data[m] = {
        'proj': np.fromfile(f'{MPATH}/pca_{MNAMES[m]}_50d_10k.bin', dtype=np.float32).reshape(-1,50),
        'mean': np.fromfile(f'{MPATH}/pca_{MNAMES[m]}_mean_10k.bin', dtype=np.float32),
        'comp': np.fromfile(f'{MPATH}/pca_{MNAMES[m]}_comp50_10k.bin', dtype=np.float32).reshape(50, N).T,
    }

def extract_road(s):
    s = s.astype(np.float64, copy=False)
    c = s - s.mean()
    mu = max(s.mean(), 1e-12)
    o = np.empty(10, dtype=np.float32)
    o[0] = np.sum(np.diff(np.sign(c)) != 0)
    o[1] = np.var(s) / mu**2
    o[2] = np.polyfit(np.arange(N), s, 1)[0] / mu
    o[3] = s[:150].mean() / mu; o[4] = s[150:300].mean() / mu
    o[5] = s[300:450].mean() / mu; o[6] = s[450:].mean() / mu
    ac = np.correlate(c, c, mode='same')
    cs = ac[300:400]
    if cs[0] > 1e-12:
        d_ = np.diff(cs); ap = 0
        for k in range(1, len(d_)):
            if d_[k-1] >= 0 and d_[k] < 0: ap = k; break
    else: ap = 0
    o[7] = ap
    o[8] = s[:150].mean() / (s[-150:].mean() + 1e-12)
    o[9] = np.sqrt(np.mean(np.diff(s)**2)) / mu
    return o

print("\n[4] 生成查询（一次性共用）...")
np.random.seed(42)
q_idx = np.random.choice(len(specs), NQ, replace=False)
q_clean = specs[q_idx].copy()
q_thick = thick[q_idx]
q_label = label[q_idx]
np.random.seed(123)
q_05 = np.clip(q_clean + np.random.randn(NQ, N) * 0.005, 0, None).astype(np.float32)
q_10 = np.clip(q_clean + np.random.randn(NQ, N) * 0.01, 0, None).astype(np.float32)
queries = [('clean', q_clean), ('0.5%', q_05), ('1.0%', q_10)]

ref_thick = {}; ref_spec = {}
for m in range(NMAT):
    mask = label == m
    ref_thick[m] = thick[mask]; ref_spec[m] = specs[mask]

# === BF-601D ===
print("\n[5] BF-601D ...")
bf = {}
for nl, qset in queries:
    pred, times = [], []
    for q in qset:
        t0 = time.perf_counter()
        idx = np.argmin(((specs - q)**2).sum(1))
        pred.append(thick[idx]); times.append(time.perf_counter()-t0)
    pred = np.array(pred); err = np.abs(pred - q_thick)
    bf[nl] = (np.mean(err<=1)*100, np.mean(times)*1e6)

# === KDT-601D ===
print("\n[6] KDT-601D ...")
t0=time.perf_counter(); kdt601=KDTree(specs, leafsize=30)
print(f'  构建: {time.perf_counter()-t0:.2f}s')
for qi in q_clean[:3]: kdt601.query(qi.reshape(1,-1), k=1)
kdt_res = {}
for nl, qset in queries:
    pred, times = [], []
    for q in qset:
        t0=time.perf_counter()
        d,ii=kdt601.query(q.reshape(1,-1),k=1)
        pred.append(thick[ii[0]]); times.append(time.perf_counter()-t0)
    pred=np.array(pred); err=np.abs(pred-q_thick)
    kdt_res[nl]=(np.mean(err<=1)*100, np.mean(times)*1e6)

# === ROAD+PCA-KDT ===
print("\n[7] ROAD+PCA-KDT ...")
kdt50 = {}
for m in range(NMAT):
    t0=time.perf_counter(); kdt50[m]=KDTree(pca_data[m]['proj'], leafsize=30)
    print(f'  {MNAMES[m]}: {time.perf_counter()-t0:.2f}s')
road_res = {}
for nl, qset in queries:
    pred_t, pred_m, times = np.zeros(NQ), np.zeros(NQ,dtype=np.int32), []
    for i,q in enumerate(qset):
        t0=time.perf_counter()
        f=extract_road(q); fn=(f-rmean)/rstd
        m=rlab[np.argmin(((rfeat-fn)**2).sum(1))]
        pred_m[i]=m
        q50=((q.astype(np.float64)-pca_data[m]['mean'])@pca_data[m]['comp']).astype(np.float32)
        d,ii=kdt50[m].query(q50.reshape(1,-1),k=10)
        d2=((ref_spec[m][ii[0]]-q)**2).sum(1)
        pred_t[i]=ref_thick[m][ii[0][np.argmin(d2)]]
        times.append(time.perf_counter()-t0)
    err=np.abs(pred_t-q_thick)
    r_acc=np.mean(pred_m==q_label)*100
    road_res[nl]=(np.mean(err<=1)*100, np.mean(times)*1e6, r_acc)

# === 汇总 ===
print("\n" + "="*70)
print("结果汇总 (10K/材料=40K, 200查询, 同一噪声)")
print("="*70)
print(f"{'噪声':>6} {'方法':>15} {'P1nm':>7} {'延迟(us)':>9} {'加速比':>8} {'路由%':>7}")
print("-"*55)
bf_cl = bf['clean'][1]
for nl in ['clean', '0.5%', '1.0%']:
    for mn in ['BF-601D', 'KDT-601D', 'ROAD+KDT']:
        if mn == 'ROAD+KDT':
            p1, lat, rt = road_res[nl]
        elif mn == 'KDT-601D':
            p1, lat = kdt_res[nl]; rt = None
        else:
            p1, lat = bf[nl]; rt = None
        spd = bf_cl/lat if lat>0 else 0
        rt_s = f"{rt:>6.1f}%" if rt is not None else "   N/A"
        print(f"{nl:>6} {mn:>15} {p1:>6.1f}% {lat:>8.0f} {spd:>7.1f}x {rt_s}")

print("\n关键对比:")
for nl in ['clean', '0.5%', '1.0%']:
    kl = kdt_res[nl][1]; rl = road_res[nl][1]
    if kl > rl:
        print(f"  {nl:>5}: KDT-601D={kl:.0f}us  ROAD+KDT={rl:.0f}us  -> ROAD 快了 {kl/rl:.1f}x")
    else:
        print(f"  {nl:>5}: KDT-601D={kl:.0f}us  ROAD+KDT={rl:.0f}us  -> KDT-601D 快了 {rl/kl:.1f}x")
print("\nDone!")
