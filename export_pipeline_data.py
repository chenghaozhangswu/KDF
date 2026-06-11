import numpy as np, sys
np.random.seed(42)
NW = 601; MAT = ['ox','sin','soi','cauthy']
BD = 'bench_data'

print("=== Exporting per-material PCA + libs ===")
lib_601d = {}; lib_norm = {}; lib_thick = {}
for name in MAT:
    raw = np.fromfile(f'{BD}/spec_{name}.bin', dtype=np.float32).reshape(-1, NW)
    t = np.fromfile(f'{BD}/thick_{name}.bin', dtype=np.float32)
    step = max(1, raw.shape[0] // 10000)
    lib_601d[name] = raw[::step].copy()
    lib_thick[name] = t[::step].copy()
    print(f"  {name}: {lib_601d[name].shape[0]} spectra")

# L2 normalize
for name in MAT:
    d = lib_601d[name].astype(np.float64)
    d_n = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    lib_norm[name] = d_n

# Per-material PCA via SVD
for name in MAT:
    d = lib_norm[name]
    mean = np.mean(d, axis=0)
    cent = d - mean
    U, S, Vt = np.linalg.svd(cent, full_matrices=False)
    comps = Vt.T  # (601, n_components)
    
    # Save as float32 .bin
    mean.astype(np.float32).tofile(f'{BD}/pca_{name}_mean.bin')
    comps[:, :50].astype(np.float32).tofile(f'{BD}/pca_{name}_comp50.bin')
    comps[:, :100].astype(np.float32).tofile(f'{BD}/pca_{name}_comp100.bin')
    lib_601d[name].tofile(f'{BD}/lib_{name}_601d.bin')
    lib_norm[name].astype(np.float32).tofile(f'{BD}/lib_{name}_n.bin')
    lib_thick[name].tofile(f'{BD}/lib_{name}_thick.bin')
    print(f"  {name}: PCA saved (50D+100D), lib saved")

# Generate 2000 queries (500 per material, 0-5% noise)
print("\n=== Generating 2000 queries ===")
queries = []; tmat = []; tthick = []
for m, name in enumerate(MAT):
    raw = np.fromfile(f'{BD}/spec_{name}.bin', dtype=np.float32).reshape(-1, NW)
    t = np.fromfile(f'{BD}/thick_{name}.bin', dtype=np.float32)
    step = max(1, raw.shape[0] // 500)
    for i in range(500):
        idx = min(i * step + step // 2, raw.shape[0] - 1)
        s = raw[idx].astype(np.float64)
        nl = np.random.uniform(0, 0.05)
        noise = np.random.normal(0, nl, NW).astype(np.float64)
        queries.append(s + s * noise)
        tmat.append(m)
        tthick.append(t[idx])
q_a = np.array(queries, dtype=np.float32)
q_n = q_a / (np.linalg.norm(q_a, axis=1, keepdims=True) + 1e-12)
q_n.astype(np.float32).tofile(f'{BD}/queries_n.bin')
np.array(tmat, dtype=np.int32).tofile(f'{BD}/queries_label.bin')
np.array(tthick, dtype=np.float32).tofile(f'{BD}/queries_thick.bin')
print(f"  Saved: queries_n.bin ({q_n.shape}), label, thick")
print("Done!")
