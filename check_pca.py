# check_pca.py
import numpy as np
mean = np.load(r'D:\kd_forest_v2\bench_data\pca_mean.npy')
comp = np.load(r'D:\kd_forest_v2\bench_data\pca_comp.npy')
print(f"mean: {mean.shape}, dtype={mean.dtype}")
print(f"comp: {comp.shape}, dtype={comp.dtype}")
print(f"mean[:5]: {mean[:5]}")
print(f"mean[-5:]: {mean[-5:]}")

# Check if 50D PCA projection of library data matches pca50d.bin
d50 = np.fromfile(r'D:\kd_forest_v2\bench_data\lib_ox_pca50d.bin', dtype=np.float32).reshape(-1, 50)
print(f"\nlib_ox_pca50d.bin: {d50.shape}")
print(f"  first row first 5: {d50[0,:5]}")

# Reconstruct: what should the PCA data look like?
# If pca_comp is (601, 50) then d50[i] = (spec[i] - mean) @ pca_comp
# Or if pca_comp is (50, 601) then d50[i] = pca_comp @ (spec[i] - mean)
# Let's figure out which by checking dimensions
if comp.shape[0] == d50.shape[1]:
    print(f"  comp is (50, 601) format")
    # Then d50[i] = comp @ (spec - mean) which gives 50D
    # Let's verify by loading one spectrum
    # Try to load spectrum lib_ox.bin...
else:
    print(f"  comp is (601, 50) format")
    # Then d50[i] = (spec - mean) @ comp (dot product)

# Let me check if pca3d files exist
import glob
files = glob.glob(r'D:\kd_forest_v2\bench_data\pca*')
print(f"\nPCA files: {files}")
