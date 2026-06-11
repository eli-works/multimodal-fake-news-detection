import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# 主色调定义
colors = {
    'forget': '#E8A598',      # 遗忘门 - 柔和红
    'input': '#A8C6A8',       # 输入门 - 柔和绿  
    'output': '#7BA7C7',      # 输出门 - 柔和蓝
    'cell_state': '#2E5C8A',  # 细胞状态 - 深蓝
    'text': '#333333'
}

# 1. 绘制细胞状态主通道（顶部粗蓝线）
ax.annotate('', xy=(10, 5), xytext=(0, 5),
            arrowprops=dict(arrowstyle='->', color=colors['cell_state'], lw=4))
ax.text(5, 5.5, '细胞状态传输通道', ha='center', fontsize=12, fontweight='bold', color=colors['cell_state'])

# 2. 三个门控单元（圆角矩形，垂直排列在下方）
gates = [
    {'name': '遗忘门', 'x': 2, 'color': colors['forget'], 'desc': '评价历史信息\n保留价值'},
    {'name': '输入门', 'x': 5, 'color': colors['input'], 'desc': '筛选新信息\n评估重要性'},
    {'name': '输出门', 'x': 8, 'color': colors['output'], 'desc': '控制对外表达\n调节输出幅度'}
]

for gate in gates:
    # 门控主体
    box = FancyBboxPatch((gate['x']-0.8, 2), 1.6, 1.5,
                         boxstyle="round,pad=0.1",
                         facecolor=gate['color'],
                         edgecolor='white',
                         linewidth=2)
    ax.add_patch(box)
    
    # 门名称
    ax.text(gate['x'], 2.75, gate['name'], ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    # 功能说明（下方小字）
    ax.text(gate['x'], 1.3, gate['desc'], ha='center', va='top',
            fontsize=9, color=colors['text'], linespacing=1.3)

# 3. 连接箭头（从门到主通道）
for x in [2, 5, 8]:
    ax.annotate('', xy=(x, 5), xytext=(x, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))

# 4. 输入/输出标注
ax.text(0, 2.75, '历史信息\n输入', ha='center', fontsize=9, color='gray')
ax.text(10, 2.75, '隐藏状态\n输出', ha='center', fontsize=9, color='gray')

# 装饰
ax.set_xlim(-1, 11)
ax.set_ylim(0, 6.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('lstm_conceptual.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()