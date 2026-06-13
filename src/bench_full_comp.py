"""
KD-Forest Comprehensive Benchmark
Methods:
1) KDF: 10D physical feature route → material PCA-50D KDT → 601D rerank
2) KDT-oracle: PCA-50D KDT within known material (no routing)
3) BF-oracle: 601D BF within known material
Conditions: library sizes [10K, 50K, 100K, 500K], noise [0%, 0.5%, 1%]
Metrics: P1nm%, MedAE, latency (μs)
"""
import numpy as np, time, os, sys, gc
from scipy.spatial import KDTree
from functools import partial

BD = r'D:\kd_forest_v2_gh\src\multi'
GD = r'D:\kd_forest_v2_gh\src\bench_data'  # main 500K libs
# Real data
bp = r'D:\kd_forest_v2_gh\src\multi'
REAL_SPEC = np.fromfile(f'{bp}/real_specs_interp.bin',dtype=np.float32).reshape(27,601)
REAL_LBL = np.fromfile(f'{bp}/real_labels.bin',dtype=np.int32)

# Materials
MNAMES=['OX','SIN','SOI','CAUTHY']
NMAT=4
WL = np.linspace(400,1000,601)

# ── Load library from benchmark data ──
def load_lib(size_per_mat, full_601d):
    """Load synthetic library: spectra + thickness labels"""
    # Use full 500K lib and downsample
    NS = 500000  # full size
    specs=[]
    thicks=[]
    for m in range(NMAT):
        mm=np.memmap(f'{BD}/lib_{MNAMES[m].lower()}_n.bin',dtype=np.float32,mode='r',shape=(NS,601))
        mt=np.memmap(f'{BD}/lib_{MNAMES[m].lower()}_t.bin',dtype=np.float32,mode='r',shape=(NS,))
        idx=np.linspace(0,NS-1,size_per_mat,dtype=int)
        specs.append(np.array(mm[idx],dtype=np.float32))
        thicks.append(np.array(mt[idx],dtype=np.float32))
        del mm,mt
    specs=np.concatenate(specs)  # (N*4, 601)
    thicks=np.concatenate(thicks)
    labels=np.repeat(np.arange(NMAT),size_per_mat)
    return specs,thicks,labels

# ── PCA components (trained on full data) ──
PCA_MEAN = np.fromfile(f'{BD}/pca_mean_601.bin',dtype=np.float32)  # (601,)
PCA_COMP = np.fromfile(f'{BD}/pca_comp_50x601.bin',dtype=np.float32).T  # (601, 50)

def pca_50d(spec):
    """Project to 50D PCA"""
    return (spec-PCA_MEAN) @ PCA_COMP  # (N, 50)

# ── ROAD features ──
RFEAT = np.fromfile(f'{BD}/route_feat_norm.bin',dtype=np.float32).reshape(-1,10)
RLBL = np.fromfile(f'{BD}/route_labels.bin',dtype=np.int32)
RMEAN = np.fromfile(f'{BD}/route_mean.bin',dtype=np.float32)
RSTD = np.fromfile(f'{BD}/route_std.bin',dtype=np.float32)

def extract_road(spec):
    """Extract 10D physical features (matches C++ exactly)"""
    s=spec.astype(np.float64, copy=False); c=s-s.mean(); mu=max(s.mean(),1e-12)
    out=np.empty(10,dtype=np.float32)
    out[0]=np.sum(np.diff(np.sign(c))!=0)
    out[1]=np.var(s)/mu**2
    out[2]=np.polyfit(np.arange(601),s,1)[0]/mu
    out[3]=s[:150].mean()/mu;out[4]=s[150:300].mean()/mu
    out[5]=s[300:450].mean()/mu;out[6]=s[450:].mean()/mu
    ac=np.correlate(c,c,mode='same');cs=ac[len(ac)//2:len(ac)//2+100];ap=0
    if cs[0]>1e-12:
        d_=np.diff(cs)
        for k in range(1,len(d_)):
            if d_[k-1]>=0 and d_[k]<0:ap=k;break
    out[7]=ap
    out[8]=s[:150].mean()/(s[-150:].mean()+1e-12)
    out[9]=np.sqrt(np.mean(np.diff(s)**2))/mu
    return out

# ── Query generators ──
def make_queries_synth(specs,thicks,labels,noise_std,Nq):
    """Sample Nq queries from library + add noise"""
    np.random.seed(42)
    idx=np.random.choice(len(specs),Nq,replace=False)
    q=np.array(specs[idx])
    if noise_std>0:
        q+=np.random.randn(*q.shape)*noise_std
    q=np.clip(q,0,None).astype(np.float32)
    return q,thicks[idx].astype(np.float32),labels[idx].astype(np.int32)

def make_queries_real():
    return REAL_SPEC, REAL_LBL

# ── Benchmark methods ──
class KDFRouter:
    """10D road feature nearest neighbor"""
    def __init__(self,rfeat,rlbl,mean,std):
        self.lib=(rfeat-mean)/std  # already normalized
        self.lbl=rlbl
    def route(self,spec):
        f=extract_road(spec)
        fn=(f-RMEAN)/RSTD
        d=np.sum((self.lib-fn)**2,1)
        return self.lbl[np.argmin(d)]

def bench_kdf(specs,thicks,labels,queries,qthick,qlbl,N_per_mat,rfeat,rlbl):
    """KDF: route + PCA-50D KDT + 601D rerank"""
    # Build material-specific KDTs on PCA-50D
    p50=pca_50d(specs)
    kdts={}
    for m in range(NMAT):
        mask=labels==m
        kdts[m]=KDTree(p50[mask])
    
    # Build reference for rerank
    ref_by_mat={}
    for m in range(NMAT):
        mask=labels==m
        ref_by_mat[m]=(specs[mask],thicks[mask])
    
    router=KDFRouter(rfeat,rlbl,RMEAN,RSTD)
    Nq=len(queries)
    pred_mat=np.zeros(Nq,dtype=np.int32)
    pred_thick=np.zeros(Nq,dtype=np.float32)
    t0=time.perf_counter()
    for i in range(Nq):
        # Route
        m=router.route(queries[i])
        pred_mat[i]=m
        # K=1 on material KDT
        d,ii=kdts[m].query(pca_50d(queries[i].reshape(1,-1)),k=1)
        pred_thick[i]=ref_by_mat[m][1][ii[0]]
    t_routing=time.perf_counter()-t0
    
    t0=time.perf_counter()
    # Rerank: check closest 601D within predicted material
    for i in range(Nq):
        m=pred_mat[i]
        ref,th=ref_by_mat[m]
        d601=np.sum((ref-queries[i])**2,1)
        best=np.argmin(d601)
        pred_thick[i]=th[best]
    t_rerank=time.perf_counter()-t0
    
    # Accuracy
    mat_acc=np.mean(pred_mat==qlbl)
    thick_err=np.abs(pred_thick-qthick)
    p1nm=np.mean(thick_err<=1.0)
    medae=np.median(thick_err)
    
    return {
        'mat_acc':mat_acc,'p1nm':p1nm,'medae':medae,
        'latency':(t_routing+t_rerank)/Nq*1e6
    }

def bench_kdt_oracle(specs,thicks,labels,queries,qthick,qlbl):
    """KDTree oracle: PCA-50D KDT within known material"""
    p50=pca_50d(specs)
    Nq=len(queries)
    pred=np.zeros(Nq,dtype=np.float32)
    t0=time.perf_counter()
    for i in range(Nq):
        m=qlbl[i]
        mask=labels==m
        d,ii=KDTree(p50[mask]).query(pca_50d(queries[i].reshape(1,-1)),k=1)
        pred[i]=thicks[mask][ii[0]]
    lat=time.perf_counter()-t0
    
    err=np.abs(pred-qthick)
    return {'p1nm':np.mean(err<=1),'medae':np.median(err),'latency':lat/Nq*1e6}

def bench_bf_oracle(specs,thicks,labels,queries,qthick,qlbl):
    """BF oracle: 601D BF within known material"""
    Nq=len(queries)
    pred=np.zeros(Nq,dtype=np.float32)
    t0=time.perf_counter()
    for i in range(Nq):
        m=qlbl[i]
        mask=labels==m
        d=np.sum((specs[mask]-queries[i])**2,1)
        pred[i]=thicks[mask][np.argmin(d)]
    lat=time.perf_counter()-t0
    
    err=np.abs(pred-qthick)
    return {'p1nm':np.mean(err<=1),'medae':np.median(err),'latency':lat/Nq*1e6}

# ── Main benchmark ──
SIZES=[10000,50000,100000,500000]  # per material
NOISES=[0.0,0.005,0.01]
NQ=200

outdir=r'D:\kd_forest_v2_gh\results'
os.makedirs(outdir,exist_ok=True)

print(f"{'Size':>8} {'Noise':>6} {'Method':>14} {'Mat%':>7} {'P1nm%':>7} {'MedAE':>7} {'Lat(us)':>8}")
print('-'*60)

for ns in SIZES:
    specs,thicks,labels=load_lib(ns,True)
    if ns>=500000:
        # downsample real queries for speed
        query_synth,qt,ql=make_queries_synth(specs,thicks,labels,noise_std=0.0,Nq=min(NQ,100))
    else:
        query_synth,qt,ql=make_queries_synth(specs,thicks,labels,noise_std=0.0,Nq=NQ)
    
    for noise in NOISES:
        # Generate noisy queries
        if noise>0:
            qn=np.array(query_synth)+np.random.randn(*query_synth.shape)*noise
            qn=np.clip(qn,0,None).astype(np.float32)
        else:
            qn=query_synth
        
        # KDF
        r=bench_kdf(specs,thicks,labels,qn,qt,ql,ns,RFEAT,RLBL)
        print(f"{ns:>8} {noise:>5.1%} {'KDF':>14} {r['mat_acc']*100:>6.1f}% {r['p1nm']*100:>6.1f}% {r['medae']:>7.2f} {r['latency']:>8.1f}")
        gc.collect()
        
        # KDT oracle (skip at 500K for speed)
        if ns<=100000:
            r=bench_kdt_oracle(specs,thicks,labels,qn,qt,ql)
            print(f"{ns:>8} {noise:>5.1%} {'KDT-oracle':>14} {'100.0':>7} {r['p1nm']*100:>6.1f}% {r['medae']:>7.2f} {r['latency']:>8.1f}")
            gc.collect()
        
        # BF oracle (skip at 500K, too slow)
        if ns<=50000:
            r=bench_bf_oracle(specs,thicks,labels,qn,qt,ql)
            print(f"{ns:>8} {noise:>5.1%} {'BF-601D':>14} {'100.0':>7} {r['p1nm']*100:>6.1f}% {r['medae']:>7.2f} {r['latency']:>8.1f}")
            gc.collect()
    
    # Real data
    if ns<=100000:
        rq,rl=make_queries_real()
        r=bench_kdf(specs,thicks,labels,rq,None,rl,ns,RFEAT,RLBL)
        print(f"{ns:>8} {'real':>6} {'KDF-real':>14} {r['mat_acc']*100:>6.1f}% {'N/A':>7} {'N/A':>7} {r['latency']:>8.1f}")
        gc.collect()
    
    print()

print("DONE")