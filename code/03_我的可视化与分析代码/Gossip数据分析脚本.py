import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ========== 强制指定中文字体文件路径 ==========
font_paths = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\msyhbd.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
    r'C:\Windows\Fonts\simkai.ttf',
]

font_prop = None
for fp in font_paths:
    try:
        font_prop = font_manager.FontProperties(fname=fp)
        print(f"成功加载字体: {fp}")
        break
    except:
        continue

if font_prop is None:
    raise FileNotFoundError("未找到可用的中文字体文件")

plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. Gossip 真实数据 ==========
max_len = np.array([128, 192, 256, 384, 512, 1024, 1920])
accuracy = np.array([0.8618, 0.8696, 0.8625, 0.8562, 0.8519, 0.8548, 0.8565])
fake_f1 = np.array([0.9182, 0.9225, 0.9179, 0.9140, 0.9108, 0.9133, 0.9142])

# ========== 2. 设置绘图环境 ==========
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 现代配色（深海蓝 + 琥珀橙）
color_acc = '#2E86AB'
color_f1 = '#F18F01'

# ========== 3. 绘制轻微线性连接（带阴影层） ==========
# 阴影层（增加立体感，alpha=0.2）
ax.plot(max_len, accuracy, '-', color=color_acc, linewidth=5,
        alpha=0.2, solid_capstyle='round', zorder=2)
ax.plot(max_len, fake_f1, '-', color=color_f1, linewidth=5,
        alpha=0.2, solid_capstyle='round', zorder=2)

# 主线条（线性连接，linewidth=2.5）
ax.plot(max_len, accuracy, '-', color=color_acc, linewidth=2.5,
        solid_capstyle='round', zorder=3)
ax.plot(max_len, fake_f1, '-', color=color_f1, linewidth=2.5,
        solid_capstyle='round', zorder=3)

# 标记点（空心圆/方块，白色填充+彩色边框，更精致）
ax.plot(max_len, accuracy, 'o', color=color_acc, markersize=10,
        markerfacecolor='white', markeredgewidth=2.5, zorder=5, label='准确率')
ax.plot(max_len, fake_f1, 's', color=color_f1, markersize=10,
        markerfacecolor='white', markeredgewidth=2.5, zorder=5, label='假新闻F1分数')

# ========== 4. 最佳点竖线（灰色虚线，不突兀） ==========
ax.axvline(x=192, color='#666666', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

# ========== 5. 坐标轴美化设置 ==========
ax.set_xlabel('max_len', fontsize=13, fontweight='bold')
ax.set_ylabel('性能指标', fontsize=13, fontweight='bold', fontproperties=font_prop)

ax.set_ylim(0.84, 0.94)
ax.set_xticks(max_len)  # 只显示实际测试点

# 刻度字体
ax.tick_params(axis='both', labelsize=11, colors='#333333')

# 只保留横向网格（更干净）
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='gray')

# 去掉上右边框（现代极简风格）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# ========== 6. 图例（右上角） ==========
legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                   shadow=False, fontsize=11, framealpha=0.95, edgecolor='gray')
for text in legend.get_texts():
    text.set_fontproperties(font_prop)

plt.tight_layout()

# ========== 7. 保存 ==========
plt.savefig('gossip_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()