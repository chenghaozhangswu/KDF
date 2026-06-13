"""
Generate per-scale PCA mean and components for all library sizes.
Computes SVD on the concatenated 4-material library for each scale.
Saves: pca_mean_{size}.bin (601 floats), pca_comp50_{size}.bin (50x601 floats)
"""
import numpy as np, os

MPATH = r'D:\kd_forest_v2_gh\src\multi'
MNS = ['ox','sin','soi','cauthy']; NMAT=4; N=601

for sz_name, sz_val in [('10k',10000), ('50k',50000), ('100k',100000), ('500k',500000)]:
    specs = np.concatenate([np.fromfile(f'{MPATH}/lib_{m}_n_{sz_name}.bin', dtype=np.float32).reshape(-1,N) for m in MNS], 0)
    print(f'{sz_name}: {specs.shape}')
    Xc = specs.astype(np.float64)
    mn = Xc.mean(0); Xc -= mn
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var10 = S[:10].sum()/S.sum()*100
    print(f'  10D variance: {var10:.1f}%')
    mn.astype(np.float32).tofile(f'{MPATH}/pca_mean_{sz_name}.bin')
    Vt[:50].astype(np.float32).tofile(f'{MPATH}/pca_comp50_{sz_name}.bin')
    print(f'  Saved pca_mean_{sz_name}.bin + pca_comp50_{sz_name}.bin')
print('Done')
