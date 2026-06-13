"""
Fig 3: Accuracy-Latency Pareto Front
Usage: python fig3_pareto.py
Output: paper/fig3_pareto.pdf, paper/fig3_pareto.png
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os

OUT = r'D:\kd_forest_v2_gh\paper'

methods = {
    'BF-601D':          (83.6, 2678, '#333333', 's', 12),
    'KDT-601D':         (67.5, 23161, '#cc0000', 'v', 10),
    'FAISS IVF(500/30)':(75.8, 12,   '#e67e22', '^', 9),
    'FAISS IVF(100/10)':(75.8, 22,   '#e67e22', '^', 9),
    'FAISS FlatL2(10D)':(60.5, 112,  '#e67e22', '^', 9),
    'CF-KD 10D+10':     (74.6, 13,   '#2980b9', 'o', 10),
    'CF-KD 10D+50':     (77.3, 30,   '#2980b9', 'o', 10),
    'CF-KD 10D+200':    (81.8, 86,   '#2980b9', 'o', 11),
    'CF-KD 10D+500':    (82.4, 278,  '#2980b9', 'o', 11),
    'CF-KD 20D+200':    (82.6, 208,  '#27ae60', 'D', 10),
}

fig, ax = plt.subplots(figsize=(10, 7))
for name, (p1, lat, c, m, s) in sorted(methods.items(), key=lambda x: x[1][1]):
    is_cfkd = 'CF-KD' in name
    ax.scatter(lat, p1, c=c, marker=m, s=s*15, label=name,
               zorder=5 if is_cfkd else 3,
               edgecolors='white' if is_cfkd else 'none', linewidths=1.5)

ax.set_xscale('log')
ax.set_xlabel('Latency per query (us, log scale)', fontsize=12)
ax.set_ylabel('P1nm Accuracy (%)', fontsize=12)
ax.set_title('Accuracy-Latency Pareto Front (50K Library, 0.5% Noise)', fontsize=13, fontweight='bold')
ax.set_xlim(5, 50000); ax.set_ylim(55, 90)
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9, ncol=2)

pareto_x = [12, 86, 208, 2678]  # 278(82.4%)被208(82.6%)支配，移除
pareto_y = [75.8, 81.8, 82.6, 83.6]
ax.plot(pareto_x, pareto_y, '--', color='#95a5a6', lw=1.5, alpha=0.5, label='Pareto front')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_pareto.pdf'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig3_pareto.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Fig 3 saved')