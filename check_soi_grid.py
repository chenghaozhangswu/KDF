"""
检查 SOI 厚度网格结构和 PCA 矩阵状态
"""
import numpy as np, os, struct, glob

DATA = R"D:\kd_forest_v2\bench_data"

# 1. SOI 厚度网格
print("=== SOI 厚度网格 ===")
p = os.path.join(DATA, "thick_soi.bin")
t = np.memmap(p, dtype=np.float32, mode='r')
n = len(t) // 2
t = t.reshape(n, 2)
print(f"  总数: {n}")
print(f"  前 20: {t[:20]}")
print(f"  top Si: min={t[:,0].min():.0f} max={t[:,0].max():.0f} unique={len(np.unique(t[:,0]))}")
print(f"  BOX:    min={t[:,1].min():.0f} max={t[:,1].max():.0f} unique={len(np.unique(t[:,1]))}")

top_unique = sorted(np.unique(t[:,0]))
print(f"  top Si 前 20 个值: {top_unique[:20]}")
# 看看是不是均匀网格
diffs = np.diff(top_unique)
print(f"  top Si 间隙: min={diffs.min():.0f} max={diffs.max():.0f} median={np.median(diffs):.0f}")

box_unique = sorted(np.unique(t[:,1]))
print(f"  BOX 前 20 个值: {box_unique[:20]}")
diffs_b = np.diff(box_unique)
print(f"  BOX 间隙: min={diffs_b.min():.0f} max={diffs_b.max():.0f} median={np.median(diffs_b):.0f}")

# 假如用 stride=50：取哪个网格点？
print(f"\n  stride=50: top Si 覆盖度:")
stride = 50
sel_idx = set(range(0, n, stride))
sel_top = set(np.round(t[list(sel_idx), 0]).astype(int))
print(f"    原 top 值: {len(top_unique)} 个, 选中: {len(sel_top)} 个 ({100*len(sel_top)/len(top_unique):.1f}%)")
# 看 500nm 是否被覆盖
if 500 in top_unique:
    print(f"    500nm 是否被选: {500 in sel_top}")
# 看被跳过的值
missing = sorted(set(top_unique) - sel_top)
print(f"    被跳过的前 10: {missing[:10]}")
print(f"    被跳过后 10: {missing[-10:]}")

# 2. PCA 矩阵存在吗？
print("\n=== PCA 状态 ===")
v2_files = glob.glob(os.path.join(DATA, "*pca*")) + glob.glob(os.path.join(DATA, "*PCA*"))
print(f"  v2 bench_data 中的 PCA 文件: {v2_files}")

# 检查 v1 项目
v1_dir = R"D:\kd_forest_project\bench_data"
if os.path.isdir(v1_dir):
    v1_files = glob.glob(os.path.join(v1_dir, "*pca*")) + glob.glob(os.path.join(v1_dir, "*PCA*"))
    print(f"  v1 bench_data 中的 PCA 文件: {v1_files}")
    # 也看看有没 pca 在别处
    v1_pca = glob.glob(os.path.join(R"D:\kd_forest_project", "**", "*pca*"), recursive=True)
    print(f"  v1 项目全目录 PCA: {v1_pca[:5]}")

# 3. 确认 stride=50 SOI 的 10000 点覆盖度（针对实测厚度）
print("\n=== 实测 SOI 覆盖度分析 ===")
# 找出所有 SOI 的 500K 个厚度中，离实测 SOI 最近的索引
# 先把 sub-sampled (stride=50) 的索引存下来
stride50_idx = np.array(sorted(range(0, n, stride)))
# 对每个实测 SOI 厚度，找库中最近的
soi_samples = [
    ("0.5umSOI", 500, 800),
    ("1umSOI", 1000, 800),
    ("2umSOI", 2000, 800),
    ("3umSOI", 3000, 1000),
    ("4umSOI", 4000, 1000),
    ("5umSOI", 5000, 1200),
    ("6umSOI", 6000, 1000),
    ("7umSOI", 7000, 800),
]
# 全库最近
for name, top_t, box_t in soi_samples:
    d = np.sqrt((t[:,0] - top_t)**2 + (t[:,1] - box_t)**2)
    best_i = np.argmin(d)
    # stride=50 集最近
    d_stride = d[stride50_idx]
    best_si = stride50_idx[np.argmin(d_stride)]
    print(f"  {name:>15}: 全库最近 idx={best_i} (top={t[best_i,0]:.0f}, box={t[best_i,1]:.0f}), stride50 最近 idx={best_si} (top={t[best_si,0]:.0f}, box={t[best_si,1]:.0f})")
