# test_deriv_route.py - Test derivative/preprocessing for real data routing
import numpy as np
from scipy.signal import savgol_filter

wl_lib = np.arange(400, 1001, dtype=float)
pca_mean = np.load(r'D:\kd_forest_v2\bench_data\pca_mean.npy').astype(np.float64)
pca_comp = np.load(r'D:\kd_forest_v2\bench_data\pca_comp.npy').astype(np.float64)

mats = ['ox','sin','soi','cauthy']

# Load raw spectral library (sample every 10000 for speed)
spec_lib = {}
thick_lib = {}
for mn in mats:
    spec = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\spec_{mn}.bin', dtype=np.float32).reshape(-1, 601)
    thick = np.fromfile(f'D:\\kd_forest_v2\\bench_data\\thick_{mn}.bin', dtype=np.float32)
    spec_lib[mn] = spec[::10000]  # 50 points per material
    thick_lib[mn] = thick[::10000]
    print(f'{mn}: {spec_lib[mn].shape} (sampled from {spec.shape[0]})')

def load_real(path):
    d = np.loadtxt(path, delimiter=',', skiprows=2)
    return np.interp(wl_lib, d[:,0], d[:,1])

# Preprocessing methods
def prep_l2(I):
    n = np.linalg.norm(I)
    return I/n if n>0 else I

def prep_deriv(I):
    dI = np.gradient(I)
    n = np.linalg.norm(dI)
    return dI/n if n>0 else dI

def prep_deriv2(I):
    d2 = np.gradient(np.gradient(I))
    n = np.linalg.norm(d2)
    return d2/n if n>0 else d2

def prep_sg(I, w=51, poly=3):
    resid = I - savgol_filter(I, w, poly)
    n = np.linalg.norm(resid)
    return resid/n if n>0 else resid

def prep_sg_norm(I, w=51, poly=3):
    """Remove envelope but keep dc: I / smooth(I)"""
    smooth = savgol_filter(I, w, poly)
    smooth[smooth < 1e-10] = 1e-10
    r = I / smooth
    return r / np.linalg.norm(r)

# Test: NONE of the methods require per-material processing, 
# just apply to raw spectrum before PCA
# But library spec needs same prep!

# Test on all real files
methods = {
    'L2 norm': prep_l2,
    '1st deriv': prep_deriv,
    '2nd deriv': prep_deriv2,
    'SG w=31': lambda I: prep_sg(I, 31, 3),
    'SG w=71': lambda I: prep_sg(I, 71, 3),
    'SG w=151': lambda I: prep_sg(I, 151, 3),
    'SG ratio w=51': lambda I: prep_sg_norm(I, 51, 3),
    'SG ratio w=101': lambda I: prep_sg_norm(I, 101, 3),
}

# Pre-process ALL library spectra with each method
print(f'\nPre-processing library spectra...')
lib_prep = {}
for mn in mats:
    lib_prep[mn] = {}
    for mname, fn in methods.items():
        lib_prep[mn][mname] = np.array([fn(spec_lib[mn][i].astype(np.float64)) for i in range(len(spec_lib[mn]))])

# Real data
real_data = [
    ('OX 100nm', load_real(r'D:\kd_forest_v2\test_data\CE\OX\OX100nm.csv'), 'ox'),
    ('OX 500nm', load_real(r'D:\kd_forest_v2\test_data\CE\OX\OX500nm.csv'), 'ox'),
    ('OX 50nm', load_real(r'D:\kd_forest_v2\test_data\CE\OX\OX50nm.csv'), 'ox'),
    ('SIN 150nm', load_real(r'D:\kd_forest_v2\test_data\CE\SIN\SIN150nm.csv'), 'sin'),
    ('SIN 500nm', load_real(r'D:\kd_forest_v2\test_data\CE\SIN\SIN500nm.csv'), 'sin'),
    ('SOI 2um', load_real(r'D:\kd_forest_v2\test_data\CE\SOI\2umSOI.csv'), 'soi'),
    ('SOI 7um', load_real(r'D:\kd_forest_v2\test_data\CE\SOI\7umSOI.csv'), 'soi'),
    ('CAUTHY 500nm', load_real(r'D:\kd_forest_v2\test_data\CE\CAUTYONGLASS\Cauthyonglass500nm.csv'), 'cauthy'),
    ('POLY 500nm', load_real(r'D:\kd_forest_v2\test_data\CE\POLY\Poly500nm.csv'), 'poly'),
]

for mname, fn in methods.items():
    print(f'\n{"="*60}')
    print(f'Method: {mname}')
    print(f'{"Sample":15s}  {"GT":8s}  {"Pred":8s}  {"Dist":10s}  {"OK":>4s}')
    print(f'{"-"*45}')
    
    correct = 0
    total = 0
    for sname, I, gt in real_data:
        I_p = fn(I)
        pca_r = fn(I)  # same
        # Actually we need pca after prep... 
        # Wait: PCA was trained on L2-normed library spectra.
        # If we use a different prep, the PCA space is different.
        # Best approach: compare in prep'd 601D space directly.
        
        # L2 distance in preprocessed 601D space (not PCA)
        best_mn = None
        best_d = float('inf')
        for mi, mn in enumerate(mats):
            for i in range(len(lib_prep[mn][mname])):
                d = np.sqrt(np.sum((I_p - lib_prep[mn][mname][i])**2))
                if d < best_d:
                    best_d = d
                    best_mn = mn
        
        ok = 'OK' if best_mn == gt else 'XX'
        if ok == 'OK': correct += 1
        total += 1
        print(f'{sname:15s}  {gt:8s}  {best_mn:8s}  {best_d:>10.6f}  {ok:>4s}')
    
    print(f'  --> {correct}/{total} = {100*correct/total:.1f}%')

print(f'\n{"="*60}')
print('DONE')
