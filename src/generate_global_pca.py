"""
generate_global_pca.py — Generate global PCA for material routing

Outputs:
  bench_data/pca_mean.npy            (601,) float32
  bench_data/pca_comp.npy            (601, 601) float32
  bench_data/pca_mean_601.bin        (601,) float32   (C++ compatible)
  bench_data/pca_comp_50x601.bin     (50, 601) float32 (C++ compatible)

Method: SVD on L2-normalized 40K sub-sampled spectra (10K per material)
from lib_*_n.bin files. These are loaded from bench_data/.
"""
import numpy as np
import os, sys

BD = os.path.join(os.path.dirname(__file__) or '.', 'bench_data')
MAT_NAMES = ['ox', 'sin', 'soi', 'cauthy']
NW = 601

def main():
    print("=== generate_global_pca.py ===\n")
    
    # 1. Load L2-normalized spectra
    print("Loading L2-normalized spectra from lib_*_n.bin...")
    all_spec = []
    for m in MAT_NAMES:
        path = os.path.join(BD, f'lib_{m}_n.bin')
        d = np.fromfile(path, dtype=np.float32).reshape(-1, NW)
        all_spec.append(d)
        print(f"  {m}: {d.shape[0]} spectra")
    
    X = np.vstack(all_spec).astype(np.float64)  # (40000, 601)
    print(f"  total: {X.shape[0]} spectra")
    
    # 2. Compute mean
    mean = np.mean(X, axis=0)
    print(f"  mean shape: {mean.shape}")
    
    # 3. SVD on centered data
    print("Computing SVD (601 x 40000)...")
    centered = X - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt: (601, 601), rows are principal components (eigenvectors)
    # C++ expects: components stored as [comp_d][wavelength_k] = row-major
    # Vt[:50] → (50, 601) — correct layout: each row is one component with 601 wavelengths
    # DO NOT use Vt.T[:, :50] — that stores as [wavelength][component] which is WRONG for C++
    comps = Vt.T.astype(np.float32)  # (601, 601), columns are PCs (for Python matmul)
    comps_cpp = Vt[:50].astype(np.float32)  # (50, 601), rows are PCs (for C++ row-major)
    mean = mean.astype(np.float32)
    print(f"  SVD done: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    
    # 4. Variance explained
    var_total = np.sum(S**2)
    var_explained = np.cumsum(S**2) / var_total
    print(f"\n  Variance explained:")
    for d in [3, 10, 20, 50]:
        print(f"    top-{d}: {var_explained[d-1]*100:.1f}%")
    
    # 5. Save as .npy
    np.save(os.path.join(BD, 'pca_mean.npy'), mean)
    np.save(os.path.join(BD, 'pca_comp.npy'), comps)
    print(f"\n  Saved: pca_mean.npy, pca_comp.npy")
    
    # 6. Save as .bin (C++ compatible)
    # C++ reads c[d * 601 + k] = component d, wavelength k
    # comps_cpp has shape (50, 601): row d = component d
    mean.tofile(os.path.join(BD, 'pca_mean_601.bin'))
    comps_cpp.tofile(os.path.join(BD, 'pca_comp_50x601.bin'))
    print(f"  Saved: pca_mean_601.bin, pca_comp_50x601.bin (C++ compatible)")
    print("\n===== DONE =====")

if __name__ == '__main__':
    main()