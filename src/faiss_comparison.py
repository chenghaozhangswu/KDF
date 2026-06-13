"""
FAISS vs CF-KD comparison on 50K library with 0.5% noise.
Requires: pip install faiss-cpu numpy scipy

Usage: python faiss_comparison.py
"""
import numpy as np, time, faiss
from scipy.spatial import KDTree

MPATH = r'D:\kd_forest_v2_gh\src\multi'
M = ['ox','sin','soi','cauthy']; NMAT=4; N=601; NQ=500; SZ=50000

print("Loading 50K library...")
specs, thick = [], []
for m in range(NMAT):
    s=np.fromfile(f'{MPATH}/lib_{M[m]}_n_50k.bin',dtype=np.float32).reshape(-1,N)
    t=np.fromfile(f'{MPATH}/lib_{M[m]}_thick_50k.bin',dtype=np.float32)
    specs.append(s); thick.append(t)
specs=np.concatenate(specs,0).astype(np.float32)
thick=np.concatenate(thick,0).astype(np.float32)

# Per-scale PCA
Xc=specs.astype(np.float64); ma=Xc.mean(0); Xc-=ma
U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
c10=Vt[:10].T.astype(np.float32)

# Queries (per-material)
np.random.seed(42)
qc=np.zeros((NMAT*NQ,N),dtype=np.float32); qt=np.zeros(NMAT*NQ)
for m in range(NMAT):
    idx=np.random.choice(SZ,NQ,replace=False)
    qc[m*NQ:(m+1)*NQ]=specs[m*SZ:(m+1)*SZ][idx]
    qt[m*NQ:(m+1)*NQ]=thick[m*SZ:(m+1)*SZ][idx]
np.random.seed(123)
qn=np.clip(qc+np.random.randn(NMAT*NQ,N)*0.005,0,None).astype(np.float32)

# PCA projection
p10=(Xc@c10.astype(np.float64)).astype(np.float32)
q10=((qn.astype(np.float64)-ma)@c10.astype(np.float64)).astype(np.float32)

# BF-601D
t1=time.perf_counter(); pr=np.zeros(NMAT*NQ)
for m in range(NMAT):
    sub=specs[m*SZ:(m+1)*SZ]
    for i in range(NQ):
        d2=((sub-qn[m*NQ+i])**2).sum(1); pr[m*NQ+i]=thick[m*SZ+np.argmin(d2)]
bl=(time.perf_counter()-t1)/(NMAT*NQ)*1e6
bp=np.mean(np.abs(pr-qt)<=1)*100
print(f"BF-601D: P1={bp:.1f}% lat={bl:.0f}us")

# FAISS IVF
for nl,nprobe in [(100,10),(500,30)]:
    qz=faiss.IndexFlatL2(10); ix=faiss.IndexIVFFlat(qz,10,nl,faiss.METRIC_L2)
    ix.train(p10); ix.add(p10); ix.nprobe=nprobe
    t1=time.perf_counter()
    D,I=ix.search(q10.astype(np.float32),50)
    ll=(time.perf_counter()-t1)/(NMAT*NQ)*1e6
    pr=np.zeros(NMAT*NQ)
    for i in range(NMAT*NQ):
        d2=((specs[I[i]]-qn[i])**2).sum(1); pr[i]=thick[I[i][np.argmin(d2)]]
    p1=np.mean(np.abs(pr-qt)<=1)*100
    print(f"FAISS IVF(nlist={nl},nprobe={nprobe}): P1={p1:.1f}% lat={ll:.0f}us")

# FAISS FlatL2 K=1
ix=faiss.IndexFlatL2(10); ix.add(p10)
t1=time.perf_counter()
D,I=ix.search(q10.astype(np.float32),1)
ll=(time.perf_counter()-t1)/(NMAT*NQ)*1e6
p1=np.mean(np.abs(thick[I.flatten()]-qt)<=1)*100
print(f"FAISS FlatL2(K=1): P1={p1:.1f}% lat={ll:.0f}us")

# CF-KD 10D+200
kdt=KDTree(p10,leafsize=30)
t1=time.perf_counter()
d,ii=kdt.query(q10.astype(np.float32),k=200)
ll=(time.perf_counter()-t1)/(NMAT*NQ)*1e6
pr=np.zeros(NMAT*NQ)
for i in range(NMAT*NQ):
    d2=((specs[ii[i]]-qn[i])**2).sum(1); pr[i]=thick[ii[i][np.argmin(d2)]]
p1=np.mean(np.abs(pr-qt)<=1)*100
print(f"CF-KD(10D+200): P1={p1:.1f}% lat={ll:.0f}us")

# CF-KD 10D+500
t1=time.perf_counter()
d,ii=kdt.query(q10.astype(np.float32),k=500)
ll=(time.perf_counter()-t1)/(NMAT*NQ)*1e6
pr=np.zeros(NMAT*NQ)
for i in range(NMAT*NQ):
    d2=((specs[ii[i]]-qn[i])**2).sum(1); pr[i]=thick[ii[i][np.argmin(d2)]]
p1=np.mean(np.abs(pr-qt)<=1)*100
print(f"CF-KD(10D+500): P1={p1:.1f}% lat={ll:.0f}us")
print("\nDone.")
