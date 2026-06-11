"""Export PCA 50D transformed route set to .bin for C++"""
import numpy as np
import os

BENCH = r'D:\kd_forest_v2\bench_data'
NW = 601

# Load PCA
print("Loading PCA ...")
pca_mean = np.load(os.path.join(BENCH, 'pca_mean.npy')).astype(np.float32)
pca_comp = np.load(os.path.join(BENCH, 'pca_comp.npy')).astype(np.float32)[:50]  # (50, 601)

# Save mean and comps as .bin (row-major)
pca_mean.tofile(os.path.join(BENCH, 'pca_mean_601.bin'))
pca_comp.tofile(os.path.join(BENCH, 'pca_comp_50x601.bin'))
print(f"  mean: {pca_mean.shape} -> pca_mean_601.bin")
print(f"  comps: {pca_comp.shape} -> pca_comp_50x601.bin")

def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, NW)

# Build and transform route set
print("\nBuilding route set (PCA 50D)...")
MAT_NAMES = ['ox','sin','soi','cauthy']
rdata = []
for m in range(4):
    raw = load_bin(os.path.join(BENCH, f'spec_{MAT_NAMES[m]}.bin'))
    if m == 2:  # SOI 2D grid
        nr, nc = 500, 1000
        for r in range(0, nr, 2):
            for c in range(0, nc, 5):
                rdata.append(raw[r * nc + c])
    else:
        for i in range(0, raw.shape[0], 100):
            rdata.append(raw[i])

rdata = np.array(rdata, dtype=np.float32)
rdata_norm = l2norm(rdata)
rdata_centered = rdata_norm - pca_mean  # subtract mean
rdata_pca = rdata_centered @ pca_comp.T  # (N, 50)
print(f"  route set: {rdata_pca.shape} -> route_pca50d.bin ({rdata_pca.nbytes/1024:.0f} KB)")

rdata_pca.tofile(os.path.join(BENCH, 'route_pca50d.bin'))

# Build labels
rlabel = np.zeros(rdata_pca.shape[0], dtype=np.int32)
# Actually generate proper labels
labels = []
for m in range(4):
    raw = load_bin(os.path.join(BENCH, f'spec_{MAT_NAMES[m]}.bin'))
    if m == 2:
        nr, nc = 500, 1000
        n_pts = (nr // 2) * (nc // 5)
    else:
        n_pts = raw.shape[0] // 100
    labels.extend([m] * n_pts)
rlabel = np.array(labels, dtype=np.int32)
rlabel.tofile(os.path.join(BENCH, 'route_labels.bin'))
print(f"  labels: {rlabel.shape} -> route_labels.bin")

print("\n===== DONE =====")
