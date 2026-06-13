# KD-Forest: High-Dimensional Spectral Matching via Coarse-to-Fine KD Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A two-stage coarse-to-fine retrieval framework for **601-dimensional optical spectrum matching**. 
Outperforms raw KD-Tree by **100–2,500× under noise** while achieving near-brute-force accuracy.

## Key Results

| Library Size | Noise | Method | P1nm Accuracy | Latency | vs KDT-601D |
|:------------:|:-----:|:------:|:-------------:|:-------:|:-----------:|
| 10K/material | 0.5% | CF-KD (10D+Top10) | **71.2%** | **9 μs** | **2,573×** |
| 50K/material | 0.5% | CF-KD (10D+Top200) | **81.7%** | **83 μs** | **279×** |
| 100K/material | 0.5% | CF-KD (10D+Top500) | **83.8%** | **273 μs** | **85×** |
| 500K/material | 0.5% | CF-KD (20D+Top500) | **83.4%** | **2.2 ms** | KDT OOM |

**KDT-601D** degrades from 29 μs (clean) → 23 ms (0.5% noise), then OOM at 500K.
**CF-KD** stays stable: 9–2,200 μs across all scales and noise levels.

## Method

```
601D Spectrum → PCA-10D Projection → KD-Tree Coarse Search (Top-K)
                                   → 601D L2 Rerank (AVX2) → Thickness
```

## File Structure

```
src/
├── bench_coarse_fine.cpp    CF-KD adaptive strategy + BF/KDT comparison
├── bench_exhaustive.cpp     dim×topN comprehensive scan
├── bench_real.cpp           27 real spectra validation
├── bench_common.h           Common header (L2 distance, data loading)
├── bench_kdf.h              KDF implementation (reference)
├── bench_hnsw.h             HNSW implementation (reference)
├── bench_kdtree.h           KD-Tree header
├── nanoflann.hpp            KD-Tree library (single header)
├── bench_full2.py           Python full pipeline benchmark
├── bench_full_comp.py       Python full comparison
├── route_strategies.py      ROAD routing analysis
├── generate_pca.py          Per-scale PCA computation
└── compile_cf.bat           MSVC compilation script
```

## Quick Start

### Prerequisites
- **C++**: MSVC 19.51+ (VS 2026) with AVX2 support
- **Python**: 3.10+ with numpy, scipy (`pip install numpy scipy`)
- **Optional**: faiss-cpu (`pip install faiss-cpu`) for FAISS comparison

### 1. Generate PCA data
```bash
cd src
python generate_pca.py
```

### 2. Compile C++ benchmarks
Open "Developer Command Prompt for VS 2026":
```bash
cd src
cl /O2 /EHsc /arch:AVX2 /std:c++17 /I. bench_coarse_fine.cpp
cl /O2 /EHsc /arch:AVX2 /std:c++17 /I. bench_exhaustive.cpp
cl /O2 /EHsc /arch:AVX2 /std:c++17 /I. bench_real.cpp
```

Or use the batch file:
```bash
compile_cf.bat
```

### 3. Run benchmarks
```bash
bench_coarse_fine.exe   # Tables 1, 6: BF/KDT + adaptive strategy
bench_exhaustive.exe    # Table 2: all dim×K combinations
bench_real.exe          # Real data validation
python bench_full2.py   # Python pipeline verification
python route_strategies.py  # ROAD routing analysis
```

## Library Data Format

Binary float32 files in `src/multi/`:
- `lib_{mat}_n_{size}.bin` — Normalized spectra, N×601 (flat)
- `lib_{mat}_thick_{size}.bin` — Thickness, N
- `pca_mean_{size}.bin` — PCA mean, 601
- `pca_comp50_{size}.bin` — PCA 50 components, 50×601
- `route_feat_norm.bin` — Routing features, 20000×10

Materials: `ox` (SiO₂), `sin` (Si₃N₄), `soi` (SOI), `cauthy` (Cauthy)
Sizes: `10k`, `50k`, `100k`, `500k` (spectra per material)

## Citation

If you use this code, please cite:
```
@article{cfkd2026,
  title={CF-KD: Coarse-to-Fine KD Retrieval for High-Dimensional Spectral Matching},
  author={Author et al.},
  journal={Optics \& Laser Technology},
  year={2026}
}
```

## License

MIT
