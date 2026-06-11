"""Real data benchmark: BF golden vs 5 methods, per-material PCA"""
import numpy as np, os, csv, time, glob, sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import cKDTree

np.set_printoptions(suppress=True, precision=4)
BD = r'D:\kd_forest_v2\bench_data'
TD = r'D:\kd_forest_v2\test_data\CE'
PCAD = r'D:\kd_forest_v2\data\pca_transformed'
WL = np.arange(400, 1001, dtype='f4')
LPM = 500_000
M = ['SiO2', 'Si3N4', 'a-Si']

# library
print('Loading library...')
lib = np.memmap(os.path.join(BD, 'lib_all_601d.bin'), dtype='f4', mode='r').reshape(-1, 601)
lib_norm = np.linalg.norm(lib, axis=1)
lib_thk = np.memmap(os.path.join(BD, 'lib_thick.bin'), dtype='f4', mode='r')
print('Library: %d x %d' % lib.shape)

# per-material PCA
def load_pca_model(mname):
    mean = np.load(os.path.join(PCAD, 'pca_50d_%s_mean.npy' % mname)).astype('f4')
    comp = np.load(os.path.join(PCAD, 'pca_50d_%s_comp.npy' % mname)).astype('f4')
    return mean, comp

print('Loading PCA models...')
pm = {}
for mname in ['sio2', 'si3n4', 'asi']:
    pm[mname] = load_pca_model(mname)

def project(q, mi):
    """Project query using material mi's PCA model"""
    mname = ['sio2','si3n4','asi'][mi]
    mean, comp = pm[mname]
    return (q - mean) @ comp.T  # -> (50,)

def project_100d(q, mi):
    mname = ['sio2','si3n4','asi'][mi]
    mean = pm[mname][0]
    comp = np.load(os.path.join(PCAD, 'pca_100d_%s_comp.npy' % mname)).astype('f4')
    return (q - mean) @ comp.T  # -> (100,)

# load precomputed PCA + KD-Trees
lib_pca50 = np.memmap(os.path.join(BD, 'lib_pca50.bin'), dtype='f4', mode='r').reshape(-1, 50)
lib_pca100 = np.memmap(os.path.join(BD, 'lib_pca100.bin'), dtype='f4', mode='r').reshape(-1, 100)
print('Building KD-Trees...')
kdt50 = [cKDTree(lib_pca50[i*LPM:(i+1)*LPM], leafsize=30) for i in range(3)]
kdt100 = [cKDTree(lib_pca100[i*LPM:(i+1)*LPM], leafsize=30) for i in range(3)]
print('  done')

# load real CSVs (OX->0=SiO2, SIN->1=Si3N4)
def read_csv(path):
    wl, val = [], []
    with open(path) as f:
        next(f)
        for row in csv.reader(f):
            if not row or len(row) < 2: continue
            try: wl.append(float(row[0])); val.append(float(row[1]))
            except: pass
    return np.array(wl,'f4'), np.array(val,'f4')

queries = []
for folder, mi in [('OX', 0), ('SIN', 1)]:
    for fn in sorted(glob.glob(os.path.join(TD, folder, '*.csv'))):
        name = os.path.splitext(os.path.basename(fn))[0]
        wl, v = read_csv(fn)
        mask = (wl >= 399) & (wl <= 1001)
        if mask.sum() < 10: continue
        f = interp1d(wl[mask], v[mask], kind='linear', bounds_error=False, fill_value=0.0)
        q = f(WL).astype('f4')
        queries.append((name, mi, q))
        print('  %s (m=%d) OK' % (name, mi))
print('\n%d queries' % len(queries))

# ═══ BF Golden Standard ═══
print('\n' + '='*60)
print('BF-601D Golden Standard')
print('='*60)
gold = []
for name, mi, q in queries:
    s = mi*LPM; e = s+LPM
    qn = np.linalg.norm(q)
    cos = lib[s:e].dot(q) / (lib_norm[s:e]*qn + 1e-15)
    bi = s + np.argmax(cos)
    gold.append((name, mi, q, lib_thk[bi], bi))
    print('  %-20s -> %8.2f nm' % (name, lib_thk[bi]))

def run(label, use_route, ndim, k_rerank):
    print('\n' + '='*60)
    print('%s (dim=%d%s, rerank=%d, route=%s)' % (label, ndim, 'D' if ndim else '', k_rerank, 'auto' if use_route else 'oracle'))
    print('='*60)
    hits = 0
    for name, mi, q, gthk, gi in gold:
        if use_route:
            # Route: project query into EACH material's PCA space, query the tree,
            # pick the material with the smallest distance to its library
            dsts = []
            for rmi in range(3):
                if ndim == 50:
                    qp = project(q, rmi)
                    d, _ = kdt50[rmi].query(qp.reshape(1,-1), 1)
                else:
                    qp = project_100d(q, rmi)
                    d, _ = kdt100[rmi].query(qp.reshape(1,-1), 1)
                dsts.append(d[0])
            rmi = np.argmin(dsts)
        else:
            rmi = mi
        
        if k_rerank == 0:
            if ndim == 50:
                qp = project(q, rmi)
                d, idx = kdt50[rmi].query(qp.reshape(1,-1), 1)
            else:
                qp = project_100d(q, rmi)
                d, idx = kdt100[rmi].query(qp.reshape(1,-1), 1)
            pi = rmi*LPM + idx[0]
        else:
            qp = project(q, rmi)
            d, idx = kdt50[rmi].query(qp.reshape(1,-1), k_rerank)
            cand = rmi*LPM + idx[0]
            qn = np.linalg.norm(q)
            cos = lib[cand].dot(q) / (lib_norm[cand]*qn + 1e-15)
            pi = cand[np.argmax(cos)]
        
        pred = lib_thk[pi]
        err = abs(pred - gthk)
        hit = (pi == gi)
        if hit: hits += 1
        flag = 'OK' if hit else 'ERR(%.2fnm)' % err
        print('  %-20s BF=%8.2f  pred=%8.2f  err=%7.2f  %s' % (name, gthk, pred, err, flag))
    print('  ==> %d/%d = %.1f%%' % (hits, len(gold), 100*hits/len(gold)))
    return hits

print('\n' + '#'*60)
print('# FIVE METHODS')
print('#'*60)
r = []
r.append(run('M1: Oracle KDT-50D',       False, 50, 0))
r.append(run('M2: Oracle KDT-100D',      False, 100, 0))
r.append(run('M3: Oracle KDF50',         False, 50, 50))
r.append(run('M4: Routed KDT-50D',       True, 50, 0))
r.append(run('M5: Routed KDF50',         True, 50, 50))

print('\n' + '#'*60)
print('# SUMMARY')
print('#'*60)
for i, (label, h) in enumerate(zip(
    ['M1: Oracle KDT-50D', 'M2: Oracle KDT-100D', 'M3: Oracle KDF50',
     'M4: Routed KDT-50D', 'M5: Routed KDF50'], r)):
    print('  %-25s %d/%d = %.1f%%' % (label, h, len(gold), 100*h/len(gold)))
