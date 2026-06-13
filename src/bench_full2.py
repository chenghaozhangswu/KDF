"""
KD-Forest Comprehensive Benchmark v2
Methods: KDF(routing+PCA50D KDT→601D rerank), KDT-oracle, BF-601D-oracle
Sizes: 10K/50K/100K per mat; Noise: 0%/0.5%/1%; Real CSV:27
"""
import numpy as np, time, os, gc, sys
from scipy.spatial import KDTree

MPATH = r'D:\kd_forest_v2_gh\src\multi'            # lib files
GPATH = r'D:\kd_forest_v2_gh\src\bench_data'         # global PCA
MNAMES = ['ox','sin','soi','cauthy']; NMAT=4; N=601
REAL_SPEC = np.fromfile(f'{MPATH}/real_specs_interp.bin',dtype=np.float32).reshape(27,601)
REAL_LBL  = np.fromfile(f'{MPATH}/real_labels.bin',dtype=np.int32)

# ── global PCA ──
GMEAN = np.fromfile(f'{GPATH}/pca_mean_601.bin',dtype=np.float32)          # (601,)
GCOMP = np.fromfile(f'{GPATH}/pca_comp_50x601.bin',dtype=np.float32).reshape(50,601).T.astype(np.float64)  # (601,50)
# GCOMP stored as 50x601; we need (601,50) for matmul: (N,601)×(601,50)

# ── route lib ──
RFEAT = np.fromfile(f'{MPATH}/route_feat_norm.bin',dtype=np.float32).reshape(-1,10)
RLBL  = np.fromfile(f'{MPATH}/route_labels.bin',dtype=np.int32)
RMEAN = np.fromfile(f'{MPATH}/route_mean.bin',dtype=np.float32)
RSTD  = np.fromfile(f'{MPATH}/route_std.bin',dtype=np.float32)

def load_lib(size):
    Nmat=size
    spec,thick,label = [],[],[]
    for m in range(NMAT):
        fn = f'{MPATH}/lib_{MNAMES[m]}_n_{size//1000}k.bin'
        ft = f'{MPATH}/lib_{MNAMES[m]}_thick_{size//1000}k.bin'
        if os.path.exists(fn):
            s=np.fromfile(fn,dtype=np.float32).reshape(-1,N)
            t=np.fromfile(ft,dtype=np.float32)
        else:
            # fallback: use memmap on 500K, downsample
            mm=np.memmap(f'{MPATH}/lib_{MNAMES[m]}_n_500k.bin',dtype=np.float32,mode='r',shape=(500000,601))
            mt=np.memmap(f'{MPATH}/lib_{MNAMES[m]}_thick_500k.bin',dtype=np.float32,mode='r',shape=(500000,))
            idx=np.linspace(0,499999,size,dtype=int)
            s=np.array(mm[idx]); t=np.array(mt[idx]); del mm,mt
        spec.append(s); thick.append(t); label.append(np.full(size,m,dtype=np.int32))
    return np.concatenate(spec,0), np.concatenate(thick,0), np.concatenate(label,0)

def extract_road(s):
    s=s.astype(np.float64,copy=False); c=s-s.mean(); mu=max(s.mean(),1e-12)
    o=np.empty(10,dtype=np.float32)
    o[0]=np.sum(np.diff(np.sign(c))!=0)
    o[1]=np.var(s)/mu**2
    o[2]=np.polyfit(np.arange(601),s,1)[0]/mu
    o[3]=s[:150].mean()/mu;o[4]=s[150:300].mean()/mu;o[5]=s[300:450].mean()/mu;o[6]=s[450:].mean()/mu
    ac=np.correlate(c,c,mode='same');cs=ac[300:400];ap=0
    if cs[0]>1e-12:
        d_=np.diff(cs);ap=0
        for k in range(1,len(d_)):
            if d_[k-1]>=0 and d_[k]<0:ap=k;break
    o[7]=ap;o[8]=s[:150].mean()/(s[-150:].mean()+1e-12)
    o[9]=np.sqrt(np.mean(np.diff(s)**2))/mu
    return o

def make_queries(specs,thicks,labels,noise,Nq):
    np.random.seed(42); idx=np.random.choice(len(specs),Nq,replace=False)
    q=np.array(specs[idx])
    if noise>0:q+=np.random.randn(*q.shape)*noise
    return np.clip(q,0,None).astype(np.float32), thicks[idx].astype(np.float32), labels[idx]

# ── benchmarks ──
def bench_kdf(specs,thicks,labels,q_query,qthick,qlbl):
    Nq=len(q_query)
    # pre-compute PCA-50D for all spectra
    p50=(specs.astype(np.float64)-GMEAN)@GCOMP.astype(np.float64); p50=p50.astype(np.float32)
    # per-material references
    ref_s={m:specs[labels==m] for m in range(NMAT)}
    ref_t={m:thicks[labels==m] for m in range(NMAT)}
    # route + KDT
    pred_mat=np.zeros(Nq,np.int32); pred_t=np.zeros(Nq,np.float32)
    t0=time.perf_counter()
    for i in range(Nq):
        f=extract_road(q_query[i]); fn=(f-RMEAN)/RSTD
        m=RLBL[np.argmin(((RFEAT-fn)**2).sum(1))]
        pred_mat[i]=m
    t_r=time.perf_counter()-t0
    # for each predicted material, NN in its PCA-50D KDT
    t0=time.perf_counter()
    kdts={}
    for m in range(NMAT):
        msk=labels==m; kdts[m]=KDTree(p50[msk])
    for i in range(Nq):
        m=pred_mat[i]; q50=(q_query[i].astype(np.float64)-GMEAN)@GCOMP
        d,ii=kdts[m].query(q50.reshape(1,-1).astype(np.float32),k=1)
        pred_t[i]=ref_t[m][ii[0]]
    t_kdt=time.perf_counter()-t0
    # rerank 601D
    t0=time.perf_counter()
    for i in range(Nq):
        m=pred_mat[i]; ref=ref_s[m]
        d=((ref-q_query[i])**2).sum(1)
        pred_t[i]=ref_t[m][np.argmin(d)]
    t_rr=time.perf_counter()-t0
    err=np.abs(pred_t-qthick)
    return {'mat%':np.mean(pred_mat==qlbl),'p1nm':np.mean(err<=1),'medae':np.median(err),
            'lat':(t_r+t_kdt+t_rr)/Nq*1e6}

def bench_kdt_oracle(specs,thicks,labels,q_query,qthick,qlbl):
    p50=(specs.astype(np.float64)-GMEAN)@GCOMP; p50=p50.astype(np.float32)
    Nq=len(q_query); pred=np.zeros(Nq,np.float32)
    ref_t={m:thicks[labels==m] for m in range(NMAT)}
    t0=time.perf_counter()
    for m in range(NMAT):
        msk=labels==m; kdts={m:KDTree(p50[msk])}
    for i in range(Nq):
        m=qlbl[i]; q50=(q_query[i].astype(np.float64)-GMEAN)@GCOMP
        d,ii=kdts[m].query(q50.reshape(1,-1).astype(np.float32),k=1)
        pred[i]=ref_t[m][ii[0]]
    lat=time.perf_counter()-t0
    err=np.abs(pred-qthick)
    return {'p1nm':np.mean(err<=1),'medae':np.median(err),'lat':lat/Nq*1e6}

def bench_bf(specs,thicks,labels,q_query,qthick,qlbl):
    Nq=len(q_query); pred=np.zeros(Nq,np.float32)
    ref_s={m:specs[labels==m] for m in range(NMAT)}
    ref_t={m:thicks[labels==m] for m in range(NMAT)}
    t0=time.perf_counter()
    for i in range(Nq):
        m=qlbl[i]; d=((ref_s[m]-q_query[i])**2).sum(1)
        pred[i]=ref_t[m][np.argmin(d)]
    lat=time.perf_counter()-t0
    err=np.abs(pred-qthick)
    return {'p1nm':np.mean(err<=1),'medae':np.median(err),'lat':lat/Nq*1e6}

# ── main ──
SIZES=[10000,50000,100000]; NOISES=[0.0,0.005,0.01]; NQ=200
OUT=r'D:\kd_forest_v2_gh\results\bench_full.txt'

with open(OUT,'w') as f:
    def log(s): print(s); f.write(s+'\n')
    log(f"{'Size':>8} {'Noise':>6} {'Method':>14} {'Mat%':>7} {'P1nm%':>7} {'MedAE':>8} {'Lat(us)':>9}")
    log('-'*62)
    
    for sz in SIZES:
        print(f'\nLoading {sz}...'); sys.stdout.flush()
        specs,thicks,labels=load_lib(sz)
        q_all,qt_all,ql_all=make_queries(specs,thicks,labels,0.0,NQ)
        
        for ns in NOISES:
            q,qt,ql=q_all,qt_all,ql_all
            if ns>0: q,qt,ql=make_queries(specs,thicks,labels,ns,NQ)
            
            r=bench_kdf(specs,thicks,labels,q,qt,ql)
            log(f"{sz:>8} {ns:>5.1%} {'KDF':>14} {r['mat%']*100:>6.1f}% {r['p1nm']*100:>6.1f}% {r['medae']:>8.2f} {r['lat']:>9.1f}")
            gc.collect()
            
            if sz<=50000:
                r=bench_bf(specs,thicks,labels,q,qt,ql)
                log(f"{sz:>8} {ns:>5.1%} {'BF-601D-oracle':>14} {'100.0':>7} {r['p1nm']*100:>6.1f}% {r['medae']:>8.2f} {r['lat']:>9.1f}")
                gc.collect()
        
        # Real CSV
        r=bench_kdf(specs,thicks,labels,REAL_SPEC,np.zeros(27),REAL_LBL)
        log(f"{sz:>8} {'real':>6} {'KDF-real':>14} {r['mat%']*100:>6.1f}% {'N/A':>7} {'N/A':>8} {r['lat']:>9.1f}")
        gc.collect()

log('\nDONE')