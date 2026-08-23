# 水波阵列控制与 FCD 波场分析系统

本项目是一个集成了硬件通信控制与流体力学计算成像的综合物理实验平台，由两大核心模块协同工作：

1. **下位机扬声器阵列控制系统**：通过串口与最多 24 通道（3 块控制板联级）的超声/扬声器阵列通信，实现波形参数（振幅、相位、周期）的精准下发。系统内置跨板绝对物理量纲的 LUT 校准映射以消除硬件容差，支持全自动声光同步定标轮播。

2. **基于 FCD 的偏折术波场分析系统**：利用迈德威视 (MindVision) 工业相机捕捉水面棋盘格形变图像 (Fast Checkerboard Demodulation)。通过特征空间 Sylvester 代数积分器、时均静态本底分离技术以及 Navier-Stokes 梯度修复算法，高精度反演水面三维高度场与二维位移场。

---

## 项目结构

```
Software_updated/
├── main_gui_circ.py          # 中央硬件控制 GUI 前端
├── fcd_gui.py                # FCD 波场解调分析 GUI 前端
├── fcd_backend_syl_circ.py   # 向后兼容桥接模块 (17行, 重导出)
├── speaker_controller.py     # 阵列串口通信后端
├── camera_controller.py      # 迈德威视工业相机驱动封装
├── mvsdk.py                  # 迈德威视官方 Python 接口 (ctypes)
│
├── backend/                  # FCD 计算引擎包
│   ├── __init__.py           # 包入口, 统一导出
│   ├── core.py               # FCDCore 核心类 (解调/积分/定标)
│   └── ui_selectors.py       # matplotlib 交互式选择器 (3 个类)
│
├── config/                   # 统一配置管理
│   ├── __init__.py
│   └── settings.py           # AppConfig - JSON 配置持久化
│
├── .gitignore
└── README.md
```

---

## 环境依赖

建议使用 **Python 3.8 及以上版本**：

```bash
pip install numpy opencv-python scipy matplotlib pyserial scikit-image
```

硬件驱动：本系统依赖迈德威视相机官方驱动。代码库中包含 Python 接口封装 `mvsdk.py`，运行前请确保操作系统中已正确安装迈德威视相机底层驱动程序。

---

## 模块说明

### 中央硬件控制前端 (`main_gui_circ.py`)

系统主控中枢，提供图形化交互界面：

- **全阵列参数矩阵**：动态生成支持单板（8 通道）至 3 板联级（24 通道）的操控矩阵，支持幅度、相位、周期与使能开关的并发下发
- **跨板 LUT 补偿引擎** (`_apply_lut_calibration`)：加载定标 JSON 后自动提取全局最大物理振幅作为绝对锚点，通过逆向非线性插值将用户输入的无量纲比例转化为精准的补偿驱动电压
- **自动定标流** (`start_carousel`)：一键启动全阵列自动定标，精准控制单通道激发、等待水面稳定、触发相机采集及数据冷却落盘的时序闭环
- **通用板卡调度** (`_for_each_board`)：统一 1 板/3 板分发模式，消除重复分支

### 阵列通信后端 (`speaker_controller.py`)

负责与下位机（Arduino / STM32 主控板）的串口指令封装与交互：

- 初始化时显式关闭硬件流控并防止触发下位机自动重启
- 实现 `write_waveform_params`、`write_channel_enables`、`save_configuration` 等底层指令的序列化与通信报文解析
- 内置硬件安全限制（周期/幅度钳位）

### 工业相机驱动 (`camera_controller.py`)

针对迈德威视 SDK 的面向对象封装：

- `_grab_loop`：内置严格的物理帧率节拍器，确保高频序列抓拍绝对均匀、不丢帧
- `wait_and_save`：将 TIFF 序列批量异步写入硬盘，确保文件名排序正确
- 支持镜头几何畸变矫正与平场矫正的硬件级配置下发

### FCD 波场解调分析 GUI (`fcd_gui.py`)

独立的波场解析上位机界面：

- 交互式选取有效像素边框
- 封装单帧波场解析、连续帧批处理、点波源测距以及斯格明子拓扑 Q 值计算等核心任务的触发入口
- 统一的路径配置持久化管理
- **输出总控开关**：可分别勾选“可视化图片输出 (jpg/png)”和“结构化原始数据输出 (CSV/JSON)”，互不干扰
- **实验驱动周期 (ms)**：填写后可记录到日志，用于后续推算驱动频率等实验设置

#### 单帧分析输出

除原有 h/u/v 矩阵 CSV 外，新增：

- `uv_mag_matrix_*.csv`：面内位移幅值
- `disp_direction_matrix_*.csv`：位移方向（归一化角相位，0~1，乘以 2π 得弧度）
- `norm_disp_u/v_matrix_*.csv`：归一化位移分量
- `water_mask_matrix_*.csv`：有效水区掩膜
- `single_parameters_*.json`：全部输入参数、mm/px 换算、载波信息等
- `Log_SingleFrame_*.txt`：详细参数、物理场统计与输出文件清单

#### 序列分析输出

每个已勾选结果类型均会按帧同步导出同名 CSV，例如：

- `hfield/hfield_000.csv` ↔ `hfield_000.jpg`
- `amplitude/Global_Amplitude_Envelope.csv` ↔ `Global_Amplitude_Envelope.jpg`
- `phase/phase_000.csv` ↔ `phase_000.jpg`
- `displacement/disp_u_000.csv`、`disp_v_000.csv`、`disp_uvmag_000.csv`、`disp_ph_norm_000.csv` ↔ `disp_000.jpg`
- `norm_disp/norm_disp_u_000.csv` 等 ↔ `norm_disp_000.jpg`
- `sz/sz_000.csv`、`s2d/s2d_sx_000.csv`、`momentum/momentum_px_000.csv`、`s3d/full_spin_sx_000.csv` 等

序列根目录还会输出 `time_array.csv`、`water_mask.csv`、`sequence_parameters_*.json` 和包含全部参数与统计的 `Log_ImageSeq_*.txt`。

> 注意：序列的逐帧 CSV 为 `像素行×像素列` 矩阵，按帧导出会产生较多文件，且数据量可能很大；建议按需勾选类型，或在仅需快速查看时关闭结构化数据输出。

### 流体力学计算引擎 (`backend/core.py`)

核心数学与物理引擎，将偏折术图像反演为水面绝对物理场：

- **智能边界与遮挡处理** (`_get_smooth_occlusion_mask`)：针对喇叭等遮挡物的拓扑处理，在频域积分前丝滑填补伪造梯度
- **Sylvester 特征空间积分** (`_fftinvgrad`)：利用有限差分矩阵对称性在特征空间进行 O(N²) 标量求解，避免传统 FFT 积分的周期性卷叠伪影
- **时均场静态本底分离** (`process_sequence`)：提取时间平均场并施加刚性 2 阶曲面拟合，剥离由相机微震动和镜头呼吸效应引发的大尺度积分漂移
- **靶向自动定标输出** (`run_calibration`)：自动搜索极值幅度中心点执行 Sine Fitting，输出 `speaker_lut_calibration.json` 补偿文件
- **结构化输出开关**：`FCDCore(..., out_plots=..., out_data=...)` 分别控制图片与 CSV/JSON 输出

### 配置管理 (`config/settings.py`)

统一的 JSON 配置持久化模块：

- `AppConfig` 类提供 `get()`/`set()`/`update()`/`save()` 接口
- 内置默认值回退机制，消除散布在 GUI 代码中的 `try/except json.dump/json.load` 样板

---

## 实验流程

1. **硬件连接**：确保控制主板通过 USB 接入，迈德威视相机正确连接至电脑

2. **操控水波阵列** (运行 `main_gui_circ.py`)：
   - 选择对应串口并连接，随后连接相机
   - 加载预先定标好的校准 JSON 文件（1~3 板），勾选"全局启用定标补正"
   - 调整幅度与相位参数，执行下发驱动指令

3. **采集光场数据**：设置相机曝光和帧率，录制稳定的水波序列

4. **反演物理场** (运行 `fcd_gui.py`)：
   - 填入基准参考图路径及形变图/序列路径
   - 选取区域裁切掉无用水槽边缘
   - 点击相应解析按钮生成带物理标尺的高度场、三维位移场及拓扑分析图

---

## 开发指南

从 `backend` 包导入核心类：

```python
from backend import FCDCore
```

也可通过桥接文件保持向后兼容：

```python
from fcd_backend_syl_circ import FCDCore
```

配置管理：

```python
from config import AppConfig
cfg = AppConfig("my_config.json")
cfg.set("key", value)
cfg.save()
```
