import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly import visualizing
from matplotlib import pyplot as plt
''' 单步的创建32个神经元
if_layer=neuron.IFNODE()
if_layer.reset()
# 循环时间设置50
T = 50
# 生成32个元素的随机张量
x = torch.rand([32]) / 8.
# 记录每个时刻的的放电量
s_list = []
# 记录每个时刻的电位
v_list = []
for t in range(T):
    s_list.append(if_layer(x).unsqueeze(0))
    v_list.append(if_layer.v.unsqueeze(0))
# 分别按照每个神经元的维度划分成组并画图
s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

figsize = (12, 8)
dpi = 200
visualizing.plot_2d_heatmap(array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
                            ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)


visualizing.plot_1d_spikes(spikes=s_list.numpy(), title='membrane sotentials', xlabel='simulating step',
                        ylabel='neuron index', figsize=figsize, dpi=dpi)

plt.show()
'''
# 多步模式下结果
if_layer = neuron.IFNode(step_mode='s')
# 每次行进八步，每个批次两组神经元，每组神经元有32个神经元
T = 8
N = 2
x_seq = torch.rand([T, N, 32])
# 生成32个张量
if_layer(torch.rand([32])/8.0)
# 将y_seq设为多步模式，每次执行八步，并用if_layer层向前传导
y_seq = functional.multi_step_forward(x_seq, if_layer)
if_layer.reset()
print(y_seq)
if_layer.step_mode = 'm'
y_seq = if_layer(x_seq)
if_layer.reset()
y_seq_np = y_seq.detach().numpy()

# 获取时间步长和批次大小
T, N, D = y_seq_np.shape

# 创建一个图形和子图
fig, axs = plt.subplots(N, 1, figsize=(12, 6), sharex=True)

# 绘制每个批次的结果
for i in range(N):
    axs[i].imshow(y_seq_np[:, i, :], aspect='auto', cmap='viridis')
    axs[i].set_title(f'Batch {i+1}')
    axs[i].set_ylabel('Time Steps')
    axs[i].set_xlabel('Features')

plt.tight_layout()
plt.show()