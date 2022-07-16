import matplotlib.pyplot as plt

plt.figure(figsize=(10,10)) # 这里建立的画布大小是5*5的，并不是坐标轴范围，使用“十字按钮”拖动你就懂了
plt.plot() # 画个只有坐标系的图

ax = plt.gca() # plt.gca( )进行坐标轴的移动
# # 获取你想要挪动的坐标轴，这里只有顶部、底部、左、右四个方向参数
ax.xaxis.set_ticks_position('bottom') #  要挪动底部的X轴，所以先目光锁定底部！

# 同理，要挪动Y轴的代码如下
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))

# 在这里，position位置参数有三种，这里用到了“按Y轴刻度位置挪动”
# 'data'表示按数值挪动，其后数字代表挪动到Y轴的刻度值
ax.spines['bottom'].set_position(('data',0))

ax.spines['top'].set_color('none') # 把上方的线设置为透明
ax.spines['right'].set_color('none') # 把右边的线也设为透明




plt.show()