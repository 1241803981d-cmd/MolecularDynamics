import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import windows
import os

# ================= 配置参数 =================
# 冷却速率 (K/ps) 与对应的 LAMMPS 文件 tag 标签
rates = [0.25, 0.5, 1.0, 2.0, 5.0]
tags = ['R0p25', 'R0p5', 'R1p0', 'R2p0', 'R5p0']

# LAMMPS timestep = 0.001 ps, ave/time = 5 -> 输出间隔为 0.005 ps
dt = 0.005  
padding_factor = 5  # 补零倍数（用于让频谱曲线更加平滑细腻）

# 存储所有冷速的频域数据，方便最后导出 CSV
output_data = {}

# 初始化画板
plt.figure(figsize=(10, 6), dpi=150)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(rates))) # 使用渐变色区分不同冷速

# 记录目标数组长度，防止由于意外中断导致不同文件的数据行数不一致
target_length = None 

for i, (rate, tag) in enumerate(zip(rates, tags)):
    filename = f'vacf_interface_{tag}.dat'
    
    if not os.path.exists(filename):
        print(f"Missing file: {filename}; skipped.")
        continue
        
    try:
        # 1. 读取 LAMMPS 输出的 VACF 数据 (跳过前两行 ave/time 的表头)
        data = np.loadtxt(filename, skiprows=2)
        vacf_total = data[:, 4]  # 第 5 列是 VACF_total
        
        # 处理可能的数组长度不一致问题（确保可以存入同一个 DataFrame）
        if target_length is None:
            target_length = len(vacf_total)
        else:
            # 截断或补齐到相同长度
            vacf_total = vacf_total[:target_length]
        
        # 2. 归一化 VACF: C(t) / C(0)
        vacf_norm = vacf_total / vacf_total[0]
        N = len(vacf_norm)
        
        # 3. 施加 Hann 窗函数 (消除两端截断带来的高频振荡伪影)
        window = windows.hann(N)
        vacf_windowed = vacf_norm * window
        
        # 4. 补零计算 (Zero-padding) 以提高频域插值密度
        N_pad = N * padding_factor
        
        # 5. 快速傅里叶变换 (FFT)
        vdos = np.abs(np.fft.rfft(vacf_windowed, n=N_pad))
        
        # 6. 计算对应的频率轴 (THz)
        freq = np.fft.rfftfreq(N_pad, d=dt)
        
        # 截取有效的频率范围 (0 ~ 30 THz)
        valid_idx = freq <= 30.0
        freq_plot = freq[valid_idx]
        vdos_plot = vdos[valid_idx]
        
        # 将 VDOS 曲线归一化到 [0, 1] 区间，便于直接对比峰值相对强度
        vdos_plot = vdos_plot / np.max(vdos_plot)
        
        # 保存频率轴（只需保存一次即可，因为所有参数一致）
        if 'Frequency_THz' not in output_data:
            output_data['Frequency_THz'] = freq_plot
            
        # 保存该冷速的 VDOS 数据
        output_data[f'VDOS_{rate}Kps'] = vdos_plot
        
        # 绘制预览图曲线
        plt.plot(freq_plot, vdos_plot, label=f'Cooling Rate: {rate} K/ps', 
                 linewidth=2, color=colors[i], alpha=0.8)
        
        print(f"Processed: {filename}")
        
    except Exception as e:
        print(f"Error while processing {filename}: {e}")

# ================= 预览图设置 =================
plt.title('Vibrational Density of States at the Ti-SiC Interface', fontsize=14, fontweight='bold')
plt.xlabel('Frequency (THz)', fontsize=12)
plt.ylabel('Normalized VDOS (a.u.)', fontsize=12)
plt.xlim(0, 25)  # 通常声子主要集中在 25 THz 以下
plt.ylim(0, 1.05)
plt.legend(frameon=True, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


# ================= 导出为绘图软件支持的 CSV =================
if output_data:
    df = pd.DataFrame(output_data)
    csv_name = 'VDOS_Data_for_Origin.csv'
    df.to_csv(csv_name, index=False)
    print(f"Wrote: {csv_name}")
else:
    print("No valid VACF data were found; no CSV file was generated.")
