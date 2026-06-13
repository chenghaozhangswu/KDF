
import numpy as np, time
from scipy.spatial import KDTree

MPATH = r'D:\kd_forest_v2_gh\src\multi'
MNAMES = ['ox','sin','soi','cauthy']; NMAT=4; N=601; NQ=200

print("="*70)
print("KD-Forest 快速验证: ROAD+50D KDT vs 原始 601D KDT")
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
    print(f'  {MNAMES[m]}: {len(s)} 条')
specs = np.concatenate(specs, 0)
thick = np.concatenate(thick, 0)
label = np.concatenate(label, 0)
print(f'  总库: {len(specs)} 条')

print("\n[2] 加载路由库...")
rfeat = np.fromfile(f'{MPATH}/route_feat_norm.bin', dtype=np.float32).reshape(-1,10)
rlab  = np.fromfile(f'{MPATH}/route_labels.bin', dtype=np.int32)
rmean = np.fromfile(f'{MPATH}/route_mean.bin', dtype=np.float32)
rstd  = np.fromfile(f'{MPATH}/route_std.bin', dtype=np.float32)
print(f'  路由库: {len(rfeat)} 条 x 10D')

print("\n[3] 加载 PCA...")
gmean = np.fromfile(f'{MPATH}/gmean_10k.bin', dtype=np.float32)
gcomp = np.fromfile(f'{MPATH}/gcomp50_10k.bin', dtype=np.float32).reshape(50, N).T.astype(np.float64)
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
    o[3] = s[:150].mean() / mu
    o[4] = s[150:300].mean() / mu
    o[5] = s[300:450].mean() / mu
    o[6] = s[450:].mean() / mu
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

print("\n[4] 生成查询...")
np.random.seed(42)
q_idx = np.random.choice(len(specs), NQ, replace=False)
q_clean = specs[q_idx].copy()
q_thick = thick[q_idx]
q_label = label[q_idx]

ref_thick = {}; ref_spec = {}
for m in range(NMAT):
    mask = label == m
    ref_thick[m] = thick[mask]
    ref_spec[m] = specs[mask]

# ---------- BF-601D ----------
print("\n[5] BF-601D 暴力扫描...")
bf = {}
for nname, qset in [('clean', q_clean),
    ('0.5%', q_clean + np.random.randn(NQ,N)*0.005),
    ('1.0%', q_clean + np.random.randn(NQ,N)*0.01)]:
    pred, times = [], []
    for q in qset:
        t0 = time.perf_counter()
        idx = np.argmin(((specs - q)**2).sum(1))
        pred.append(thick[idx]); times.append(time.perf_counter()-t0)
    pred = np.array(pred); err = np.abs(pred - q_thick)
    bf[nname] = (np.mean(err<=1)*100, np.mean(times)*1e6)
    print(f'  {nname:>5}: P1nm={bf[nname][0]:.1f}%  lat={bf[nname][1]:.1f}us')

# ---------- KDT-601D ----------
print("\n[6] KDT-601D 构建...")
t0=time.perf_counter(); kdt601=KDTree(specs, leafsize=30); print(f'  构建: {time.perf_counter()-t0:.2f}s')
# warmup
for qi in q_clean[:5]: kdt601.query(qi.reshape(1,-1), k=1)

kdt601_results = {}
for nname, qset in [('clean', q_clean),
    ('0.5%', q_clean + np.random.randn(NQ,N)*0.005),
    ('1.0%', q_clean + np.random.randn(NQ,N)*0.01)]:
    pred, times = [], []
    for q in qset:
        t0=time.perf_counter()
        d,ii=kdt601.query(q.reshape(1,-1),k=1)
        pred.append(thick[ii[0]]); times.append(time.perf_counter()-t0)
    pred=np.array(pred); err=np.abs(pred-q_thick)
    kdt601_results[nname]=(np.mean(err<=1)*100, np.mean(times)*1e6)
    print(f'  {nname:>5}: P1nm={kdt601_results[nname][0]:.1f}%  lat={kdt601_results[nname][1]:.1f}us')

# ---------- ROAD+PCA-KDT ----------
print("\n[7] ROAD+PCA-KDT 构建...")
kdt50 = {}
for m in range(NMAT):
    t0=time.perf_counter(); kdt50[m]=KDTree(pca_data[m]['proj'], leafsize=30)
    print(f'  {MNAMES[m]}: {time.perf_counter()-t0:.2f}s')

road_results = {}
for nname, qset in [('clean', q_clean),
    ('0.5%', q_clean + np.random.randn(NQ,N)*0.005),
    ('1.0%', q_clean + np.random.randn(NQ,N)*0.01)]:
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
    road_results[nname]=(np.mean(err<=1)*100, np.mean(times)*1e6, r_acc)
    print(f'  {nname:>5}: P1nm={road_results[nname][0]:.1f}%  route={r_acc:.1f}%  lat={road_results[nname][1]:.1f}us')

# ---------- 汇总 ----------
print("\n" + "="*70)
print("结果汇总 (10K/材料=40K, 200查询)")
print("="*70)
print(f"{'噪声':>6} {'方法':>15} {'P1nm':>7} {'延迟(us)':>10} {'加速比':>7} {'路由%':>7}")
print("-"*55)
bf_cl = bf['clean'][1]
for nl in ['clean', '0.5%', '1.0%']:
    for mn, res in [('BF-601D', bf[nl]), ('KDT-601D', kdt601_results[nl]), ('ROAD+KDT', road_results[nl])]:
        if mn == 'ROAD+KDT':
            p1, lat, rt = res
            spd = bf_cl/lat
            print(f"{nl:>6} {mn:>15} {p1:>6.1f}% {lat:>9.1f} {spd:>6.1f}x {rt:>6.1f}%")
        else:
            p1, lat = res
            spd = bf_cl/lat
            print(f"{nl:>6} {mn:>15} {p1:>6.1f}% {lat:>9.1f} {spd:>6.1f}x {'N/A':>7}")

print("\n关键对比:")
for nl in ['clean', '0.5%', '1.0%']:
    kl = kdt601_results[nl][1]
    rl = road_results[nl][1]
    if kl > rl:
        print(f"  {nl:>5}: KDT-601D={kl:.0f}us  ROAD+KDT={rl:.0f}us  -> ROAD 快了 {kl/rl:.1f}x")
    else:
        print(f"  {nl:>5}: KDT-601D={kl:.0f}us  ROAD+KDT={rl:.0f}us  -> KDT-601D 快了 {rl/kl:.1f}x")
print("\nDone!")
