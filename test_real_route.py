# test_real_route.py - Test 10D PCA route on real CSV data
import numpy as np, os, glob, re, json

def load_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=2)
    return data[:,0], data[:,1]

# List real CSV files with known material + thickness
mats = {'OX': [], 'SIN': [], 'SOI': [], 'POLY': [], 'CAUTHY': []}
mat_map = {'OX':'OX','OXIDE':'OX','SIN':'SIN','SOI':'SOI',
           'POLY':'POLY','CAUT':'CAUTHY','CAUTHY':'CAUTHY'}
base = r'D:\kd_forest_v2\test_data\CE'
for dirname in sorted(os.listdir(base)):
    dirpath = os.path.join(base, dirname)
    if not os.path.isdir(dirpath): continue
    key = dirname.upper()[:4]
    for k in mats:
        if key.startswith(k[:4]) or k[:4].startswith(key):
            for fn in sorted(glob.glob(os.path.join(dirpath, '*.csv'))):
                fname = os.path.basename(fn)
                nums = re.findall(r'(\d+\.?\d*)', fname)
                if nums:
                    mats[k].append((fn, float(nums[0])))
            break

print("=== Real CSV files ===")
real_files = []
gt_material = []
gt_thick = []
for mk in ['OX','SIN','SOI','CAUTHY','POLY']:
    for fn, t in mats.get(mk, []):
        fname = os.path.basename(fn)
        real_files.append(fn)
        gt_material.append(mk)
        gt_thick.append(t)
        print(f"  {fname:30s}  mat={mk:6s}  thick={t}nm")

print(f"\nTotal: {len(real_files)} files")
print(f"Materials: {set(gt_material)}")

# Build 10D route tree (from library subsample)
print("\n=== Loading library and building route trees ===")
lib_wl = np.arange(400, 1001, dtype=float)  # 601 points

# Load full library vectors (50D)
NM = 4
mat_names = ['ox','sin','soi','cauthy']
full50 = {}
for mn in mat_names:
    full50[mn] = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\lib_{mn}_pca50d.bin', 
                             dtype=np.float32).reshape(-1, 50)
    print(f"  {mn}: {full50[mn].shape}")

# PCA model (full 601 comps, take first 50)
pca_mean = np.load(r'D:\kd_forest_v2\bench_data\pca_mean.npy').astype(np.float64)
pca_comp = np.load(r'D:\kd_forest_v2\bench_data\pca_comp.npy').astype(np.float64)  # (601, 601)
print(f"  PCA mean: {pca_mean.shape}, PCA comp: {pca_comp.shape}")
print(f"  PCA comp[:,:50]: {pca_comp[:,:50].shape} (first 50D)")

# Build 10D route trees
import scipy.spatial

def build_route10(stride=200):
    """Build 10D route tree from subsampled library"""
    X_train, y_train = [], []
    for mi, name in enumerate(mat_names):
        data = full50[name]  # (500000, 50)
        sub = data[::stride, :10]  # first 10 dims
        X_train.append(sub)
        y_train.append(np.full(len(sub), mi, dtype=np.int32))
    X = np.concatenate(X_train, axis=0)
    y = np.concatenate(y_train, axis=0)
    tree = scipy.spatial.KDTree(X, leafsize=30)
    return tree, y

tree10s200, y10s200 = build_route10(200)
tree10s50, y10s50 = build_route10(50)

# Build 50D route tree (reference)
def build_route50(stride=200):
    X_train, y_train = [], []
    for mi, name in enumerate(mat_names):
        data = full50[name]
        sub = data[::stride]
        X_train.append(sub)
        y_train.append(np.full(len(sub), mi, dtype=np.int32))
    X = np.concatenate(X_train, axis=0)
    y = np.concatenate(y_train, axis=0)
    tree = scipy.spatial.KDTree(X, leafsize=30)
    return tree, y

tree50s200, y50s200 = build_route50(200)

# Interpolate real CSVs to library wavelength grid and normalize
def process_spectrum(wl, I, target_wl):
    """Interpolate to target grid and L2-normalize"""
    I_interp = np.interp(target_wl, wl, I)
    norm = np.linalg.norm(I_interp)
    if norm > 0:
        I_interp = I_interp / norm
    return I_interp

def pca_project(I_601, mean, comp, ndim=50):
    """Project 601-D spectrum to ndim PCA space
    comp is (601, 601) from SVD, use first ndim cols
    """
    return (I_601 - mean) @ comp[:, :ndim]  # returns (ndim,)

# Process all real files
print("\n=== Processing real spectra and testing route ===")
results = []
for i, fn in enumerate(real_files):
    wl, I = load_csv(fn)
    spec = process_spectrum(wl, I, lib_wl)
    
    # PCA to 50D and take first 10 for 10D route
    pca50 = pca_project(spec, pca_mean, pca_comp, ndim=50)
    pca10 = pca50[:10].copy()
    pca10 = pca50[:10].copy()
    
    # 10D route (stride 200)
    dist10, idx10 = tree10s200.query(pca10.reshape(1, -1), k=1)
    route10_200 = mat_names[y10s200[idx10[0]]]
    
    # 10D route (stride 50 → 40K)
    dist10_50, idx10_50 = tree10s50.query(pca10.reshape(1, -1), k=1)
    route10_50 = mat_names[y10s50[idx10_50[0]]]
    
    # 50D route (stride 200, reference)
    dist50, idx50 = tree50s200.query(pca50.reshape(1, -1), k=1)
    route50_200 = mat_names[y50s200[idx50[0]]]
    
    # Full BF search (50D, 2M points) as ground truth
    best_mat = None
    best_dist = float('inf')
    for mn in mat_names:
        d = np.min(np.sum((full50[mn] - pca50.reshape(1, -1))**2, axis=1))
        if d < best_dist:
            best_dist = d
            best_mat = mn
    
    gt = gt_material[i].lower()
    results.append({
        'file': os.path.basename(fn),
        'gt': gt,
        'gt_thick': gt_thick[i],
        'bf': best_mat,
        'bf_dist': best_dist,
        'route10_200': route10_200,
        'route10_50': route10_50,
        'route50_200': route50_200,
    })
    
    ok10_200 = 'OK' if route10_200 == gt else 'XX'
    ok10_50 = 'OK' if route10_50 == gt else 'XX'
    ok50 = 'OK' if route50_200 == gt else 'XX'
    ok_bf = 'OK' if best_mat == gt else 'XX'
    
    fname = os.path.basename(fn)[:30]
    print(f"  {fname:30s}  GT={gt:6s}  {gt_thick[i]:>6.0f}nm  "
          f"10D200={route10_200:6s}{ok10_200}  10D50={route10_50:6s}{ok10_50}  "
          f"50D={route50_200:6s}{ok50}  BF={best_mat:6s}{ok_bf}")

# Summary
print("\n=== Summary ===")
for label, key in [("10D route (stride 200)", "route10_200"),
                   ("10D route (stride 50)", "route10_50"),
                   ("50D route (stride 200)", "route50_200"),
                   ("Full BF search", "bf")]:
    correct = sum(1 for r in results if r[key] == r['gt'])
    total = len(results)
    print(f"  {label:30s}: {correct}/{total} = {100*correct/total:.1f}%")

# Full BF search as reference (not including POLY)
print("\n=== Material breakdown ===")
for mk in ['ox','sin','soi','cauthy']:
    subset = [r for r in results if r['gt'] == mk]
    if not subset: continue
    for label, key in [("10D200","route10_200"),("10D50","route10_50"),("50D","route50_200"),("BF","bf")]:
        corr = sum(1 for r in subset if r[key] == r['gt'])
        print(f"  {mk:6s} {label:6s}: {corr}/{len(subset)} = {100*corr/len(subset):.1f}%")

# Also check: are there POLY files? If so, check if they get misclassified
poly_files = mats.get('POLY', [])
if poly_files:
    print(f"\n=== POLY analysis ===")
    print(f"  POLY files found: {len(poly_files)}")
    for fn, t in poly_files:
        wl, I = load_csv(fn)
        spec = process_spectrum(wl, I, lib_wl)
        pca50 = pca_project(spec, pca_mean, pca_comp)
        pca10 = pca50[:10]
        
        # Route results
        dist, idx = tree10s200.query(pca10.reshape(1,-1), k=1)
        r10 = mat_names[y10s200[idx[0]]]
        dist, idx = tree50s200.query(pca50.reshape(1,-1), k=1)
        r50 = mat_names[y50s200[idx[0]]]
        
        # BF
        best_mn, best_d = None, float('inf')
        for mn in mat_names:
            d = np.min(np.sum((full50[mn] - pca50.reshape(1,-1))**2, axis=1))
            if d < best_d:
                best_d = d; best_mn = mn
        print(f"  {os.path.basename(fn):30s} GT=POLY  thick={t}nm  "
              f"10D={r10}  50D={r50}  BF={best_mn}  bf_dist={best_d:.6f}")
