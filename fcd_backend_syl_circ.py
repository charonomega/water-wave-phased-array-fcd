import os
import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.signal import hilbert
from datetime import datetime
import math
from scipy.signal import hilbert, butter, filtfilt # 🌟 新增滤波
from scipy.optimize import curve_fit               # 🌟 新增曲线拟合
from scipy.signal.windows import tukey

# 强行切换到 Agg 后端，确保多线程下终端不卡死
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.path import Path 

import json
from scipy.signal import hilbert
import matplotlib.patches as patches

import time
import matplotlib.lines as mlines

# 100% 确保支持中文字符与负号渲染
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False 

# =====================================================================
# 🌟 圆形阵列专属交互类 1：动态三同心圆生成器 (自动定标版)
# =====================================================================
class MasterCircleSelector:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.pts = []
        self.center = None
        self.radius_orig = None
        self.radius_target = None
        self.radius_inner = None
        self.state = 'click3'
        self.bg = None
        
        self.circle_orig = None
        self.circle_target = None
        self.circle_inner = None
        self.temp_markers, = self.ax.plot([], [], 'ro', markersize=6)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ax.set_title("【定标 1/2: 圆形阵列源圆周】\n依次点击 3 个点锁定最外侧喇叭所在的第一参考圆", color='cyan', weight='bold')

    def capture_bg(self):
        artists = [self.circle_orig, self.circle_target, self.circle_inner, self.temp_markers]
        states = [a.get_visible() if a else False for a in artists]
        for a in artists: 
            if a: a.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        for a, v in zip(artists, states): 
            if a: a.set_visible(v)

    def _fit_circle(self):
        if len(self.pts) < 3: return None
        x1, y1 = self.pts[0]; x2, y2 = self.pts[1]; x3, y3 = self.pts[2]
        A = x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2
        if abs(A) < 1e-5: return None
        B = (x1**2 + y1**2)*(y3 - y2) + (x2**2 + y2**2)*(y1 - y3) + (x3**2 + y3**2)*(y2 - y1)
        C = (x1**2 + y1**2)*(x2 - x3) + (x2**2 + y2**2)*(x3 - x1) + (x3**2 + y3**2)*(x1 - x2)
        xc, yc = -B / (2*A), -C / (2*A)
        R = np.hypot(x1 - xc, y1 - yc)
        return (xc, yc), R

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3: # 右键重画
            self.pts = []
            self.state = 'click3'
            if self.circle_orig: self.circle_orig.remove(); self.circle_orig = None
            if self.circle_target: self.circle_target.remove(); self.circle_target = None
            if self.circle_inner: self.circle_inner.remove(); self.circle_inner = None
            self.temp_markers.set_data([], [])
            self.bg = None
            self.fig.canvas.draw_idle()
            return
        
        if event.button == 1:
            if self.state == 'click3':
                self.pts.append((event.xdata, event.ydata))
                self.temp_markers.set_data([p[0] for p in self.pts], [p[1] for p in self.pts])
                self.fig.canvas.draw_idle()
                if len(self.pts) == 3:
                    res = self._fit_circle()
                    if res:
                        self.center, self.radius_orig = res
                        self.radius_target = self.radius_orig
                        self.radius_inner = self.radius_orig
                        self.circle_orig = patches.Circle(self.center, self.radius_orig, color='cyan', fill=False, lw=1.5, ls='--')
                        self.circle_target = patches.Circle(self.center, self.radius_target, color='green', fill=False, lw=2, ls='-')
                        self.circle_inner = patches.Circle(self.center, self.radius_inner, color='magenta', fill=False, lw=1.5, ls='--')
                        self.ax.add_patch(self.circle_orig)
                        self.ax.add_patch(self.circle_target)
                        self.ax.add_patch(self.circle_inner)
                        self.state = 'done'
                        self.ax.set_title("源圆周已锁定！请【按住左键】拖拽生成目标与对称参考圆，按[Enter]确认", color='green', weight='bold')
                        self.fig.canvas.draw_idle()
            elif self.state == 'done':
                self.state = 'resize'
                self.capture_bg()

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.state == 'resize' and self.center is not None:
            self.radius_target = np.hypot(event.xdata - self.center[0], event.ydata - self.center[1])
            self.radius_inner = abs(2 * self.radius_target - self.radius_orig)
            
            self.circle_target.set_radius(self.radius_target)
            self.circle_inner.set_radius(self.radius_inner)
            
            self.fig.canvas.restore_region(self.bg)
            self.ax.draw_artist(self.circle_orig)
            self.ax.draw_artist(self.circle_target)
            self.ax.draw_artist(self.circle_inner)
            self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if self.state == 'resize':
            self.state = 'done'
            self.bg = None

    def on_key(self, event):
        if event.key == 'enter' and self.state == 'done':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

# =====================================================================
# 🌟 直线阵列专属交互类：动态三平行线生成器 (自动定标版)
# =====================================================================
class MasterLineSelector:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.start_pt = None
        self.end_pt = None
        self.state = 'draw'
        self.bg = None 
        
        self.press_x = None
        self.press_y = None
        self.orig_xdata, self.orig_ydata = None, None
        self.target_xdata, self.target_ydata = None, None
        self.sym_xdata, self.sym_ydata = None, None
        
        self.line_orig, = self.ax.plot([], [], color='cyan', lw=1.5, ls='--')
        self.line_target, = self.ax.plot([], [], color='green', lw=2, ls='-')
        self.line_sym, = self.ax.plot([], [], color='magenta', lw=1.5, ls='--')
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ax.set_title("【定标 1/2: 直线阵列源基准线】\n请拖拽画出第一条紧贴喇叭阵列的源基准线", color='cyan', weight='bold')

    def capture_bg(self):
        vis1 = self.line_orig.get_visible()
        vis2 = self.line_target.get_visible()
        vis3 = self.line_sym.get_visible()
        self.line_orig.set_visible(False)
        self.line_target.set_visible(False)
        self.line_sym.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.line_orig.set_visible(vis1)
        self.line_target.set_visible(vis2)
        self.line_sym.set_visible(vis3)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3: # 右键重画
            self.start_pt = None; self.end_pt = None; self.state = 'draw'
            self.line_orig.set_data([], []); self.line_target.set_data([], []); self.line_sym.set_data([], [])
            self.bg = None; self.fig.canvas.draw_idle()
            return
        if event.button == 1:
            if self.state == 'draw':
                if self.start_pt is None:
                    self.start_pt = (event.xdata, event.ydata)
                    self.line_orig.set_data([event.xdata], [event.ydata])
                    self.capture_bg()
                else:
                    self.end_pt = (event.xdata, event.ydata)
                    self.line_orig.set_data([self.start_pt[0], self.end_pt[0]], [self.start_pt[1], self.end_pt[1]])
                    self.orig_xdata = list(self.line_orig.get_xdata())
                    self.orig_ydata = list(self.line_orig.get_ydata())
                    self.target_xdata, self.target_ydata = self.orig_xdata, self.orig_ydata
                    self.sym_xdata, self.sym_ydata = self.orig_xdata, self.orig_ydata
                    self.line_target.set_data(self.target_xdata, self.target_ydata)
                    self.line_sym.set_data(self.sym_xdata, self.sym_ydata)
                    self.state = 'done'
                    self.bg = None
                    self.ax.set_title("源基准线已锁定！请【按住左键拖拽】生成目标与对称参考线，按[Enter]确认", color='green', weight='bold')
                    self.fig.canvas.draw_idle()
            elif self.state == 'done':
                self.state = 'pan'
                self.press_x, self.press_y = event.xdata, event.ydata
                self.capture_bg()

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.bg is None: self.capture_bg()

        if self.state == 'draw' and self.start_pt is not None:
            self.line_orig.set_data([self.start_pt[0], event.xdata], [self.start_pt[1], event.ydata])
        elif self.state == 'pan' and self.press_x is not None:
            dx, dy = event.xdata - self.press_x, event.ydata - self.press_y
            self.target_xdata = [self.orig_xdata[0]+dx, self.orig_xdata[1]+dx]
            self.target_ydata = [self.orig_ydata[0]+dy, self.orig_ydata[1]+dy]
            self.sym_xdata = [self.orig_xdata[0]+2*dx, self.orig_xdata[1]+2*dx]
            self.sym_ydata = [self.orig_ydata[0]+2*dy, self.orig_ydata[1]+2*dy]
            
            self.line_target.set_data(self.target_xdata, self.target_ydata)
            self.line_sym.set_data(self.sym_xdata, self.sym_ydata)
        else:
            return
        
        self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.line_orig)
        self.ax.draw_artist(self.line_target)
        self.ax.draw_artist(self.line_sym)
        self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if self.state == 'pan':
            self.state = 'done'
            self.bg = None

    def on_key(self, event):
        if event.key == 'enter' and self.state == 'done':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

class InteractiveMeasurer:
    def __init__(self, ax, fig, cm_per_pixel):
        self.ax = ax
        self.fig = fig
        self.cm_per_pixel = cm_per_pixel

        # 🌟 终极修复核心 1：在初始状态下，死死记住当前图像最纯净的画幅边界坐标限制
        self.orig_xlim = self.ax.get_xlim()
        self.orig_ylim = self.ax.get_ylim()

        # 全局控制
        self.mode = 'p2p' # 'p2p' (点对点) 或 'pl' (平行线)
        self.state = 'idle'
        self.count = 0
        self.measurements = []
        self.bg = None
        self.ext_len = 8000 # 平行线无限延长长度
        
        # 预定义高对比度颜色表，用于固化不同的测量组
        self.colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', '#FF9900', '#FF0000']

        # ================= 动态绘图对象初始化 =================
        # 1. P2P 动态对象
        self.dyn_p2p_line, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_p2p_p1, = self.ax.plot([], [], 'wo', markersize=6, visible=False)
        self.dyn_p2p_p2, = self.ax.plot([], [], 'wo', markersize=6, visible=False)
        self.dyn_p2p_txt = self.ax.text(0, 0, "", color='black', fontsize=10, 
                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)

        # 2. PL 平行线动态对象
        self.dyn_pl_l1, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_pl_l2, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_pl_perp, = self.ax.plot([], [], color='white', lw=2, ls='-', visible=False) # 垂线
        self.dyn_pl_p1, = self.ax.plot([], [], 'ws', markersize=6, visible=False) # 垂线起点
        self.dyn_pl_p2, = self.ax.plot([], [], 'ws', markersize=6, visible=False) # 垂线终点
        self.dyn_pl_txt = self.ax.text(0, 0, "", color='black', fontsize=10,
                                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)

        # 挂载事件
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # 强制绘制一次以获取纯净背景
        self.fig.canvas.draw()
        self.capture_bg()
        self.update_title()

    def update_title(self):
        base_title = f"【按 M 键切换模式】当前: {'点对点测距' if self.mode == 'p2p' else '平行线测距'} | 已测 {self.count} 组数据\n"
        
        if self.mode == 'p2p':
            if self.state == 'idle':
                t2 = "【左键单击】确定起点"
            elif self.state == 'p1_selected':
                t2 = "【移动】拉出连线，【左键单击】锁定终点并保存"
        else: # PL 模式
            if self.state == 'idle':
                t2 = "【左键单击】确定基准线起点"
            elif self.state == 'ref_start':
                t2 = "【移动】拉出基准线对齐波前，【左键单击】锁定"
            elif self.state == 'parallel':
                t2 = "【移动】生成平行游标，【左键单击】放置"
            elif self.state in ['adjust', 'pan_l1', 'pan_l2']:
                t2 = "【左键拖拽】任一直线可平移微调 | 【C 键】确认保存 | 【右键】撤销重画"

        self.ax.set_title(base_title + t2, fontsize=11, weight='bold', color='yellow' if self.mode=='p2p' else 'cyan',
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3))
        self.fig.canvas.draw_idle()

    def capture_bg(self):
        """隐蔽所有动态元素，拍摄当前包含已固化测量的背景"""
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        vis_states = [a.get_visible() for a in artists]
        for a in artists: a.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        for a, v in zip(artists, vis_states): a.set_visible(v)

    def reset_dynamic_state(self):
        """清空未完成的测量状态，隐藏动态游标"""
        self.state = 'idle'
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        for a in artists: a.set_visible(False)
        self.capture_bg()
        self.update_title()

    def commit_measurement(self):
        """将当前满意的动态游标固化为静态图层，存入日志"""
        color = self.colors[self.count % len(self.colors)]
        
        if self.mode == 'p2p':
            p1, p2 = self.p2p_p1, self.p2p_p2
            dist_cm = np.hypot(p2[0]-p1[0], p2[1]-p1[1]) * self.cm_per_pixel
            # 绘制静态对象
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o', color=color, markersize=5)
            cx, cy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            self.ax.text(cx, cy, f" #{self.count+1}: {dist_cm:.2f}cm ", color='white', 
                         fontsize=10, ha='center', va='center', bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
            self.measurements.append({'id': self.count+1, 'type': 'P2P', 'p1': p1, 'p2': p2, 'dist_cm': dist_cm, 'time': time.strftime("%H:%M:%S")})

        elif self.mode == 'pl':
            p_start, p_end, dist_cm = self._calc_pl_geometry()
            dx, dy = self.pl_dir
            norm = np.hypot(dx, dy)
            ux, uy = dx/norm, dy/norm
            # 延长线
            l1_x = [p_start[0] - ux*self.ext_len, p_start[0] + ux*self.ext_len]
            l1_y = [p_start[1] - uy*self.ext_len, p_start[1] + uy*self.ext_len]
            l2_x = [p_end[0] - ux*self.ext_len, p_end[0] + ux*self.ext_len]
            l2_y = [p_end[1] - uy*self.ext_len, p_end[1] + uy*self.ext_len]
            
            # 绘制静态对象
            self.ax.plot(l1_x, l1_y, color=color, lw=1.5, ls='--')
            self.ax.plot(l2_x, l2_y, color=color, lw=1.5, ls='--')
            self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color, lw=2.5) # 加粗垂线
            self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 's', color=color, markersize=5)
            
            cx, cy = (p_start[0]+p_end[0])/2, (p_start[1]+p_end[1])/2
            self.ax.text(cx, cy, f" #{self.count+1}: {dist_cm:.2f}cm ", color='white', 
                         fontsize=10, ha='center', va='center', bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
            self.measurements.append({'id': self.count+1, 'type': 'Parallel', 'p1': p_start, 'p2': p_end, 'dist_cm': dist_cm, 'time': time.strftime("%H:%M:%S")})

        self.count += 1
        
        # 🌟 终极修复核心 2：在固化创建完图层后，强行把画幅拉回初始状态，遏制 Matplotlib 的乱缩放 Bug
        self.ax.set_xlim(self.orig_xlim)
        self.ax.set_ylim(self.orig_ylim)
        
        self.reset_dynamic_state() # 重置并把刚刚画的静态内容收入 Background 中

    # ================= 核心几何计算 =================
    def _calc_pl_geometry(self):
        dx, dy = self.pl_dir
        A, B = -dy, dx
        norm2 = A**2 + B**2
        if norm2 == 0: return self.pl_base, self.pl_base, 0
        
        C2 = dy * self.pl_l2_pt[0] - dx * self.pl_l2_pt[1]
        x0, y0 = self.pl_base
        x_int = x0 - A * (A * x0 + B * y0 + C2) / norm2
        y_int = y0 - B * (A * x0 + B * y0 + C2) / norm2
        
        dist_px = np.hypot(x_int - x0, y_int - y0)
        return (x0, y0), (x_int, y_int), dist_px * self.cm_per_pixel

    def _get_extended(self, pt, dx, dy):
        norm = np.hypot(dx, dy)
        if norm == 0: return [pt[0], pt[0]], [pt[1], pt[1]]
        ux, uy = dx/norm, dy/norm
        return [pt[0] - ux*self.ext_len, pt[0] + ux*self.ext_len], [pt[1] - uy*self.ext_len, pt[1] + uy*self.ext_len]

    # ================= 交互事件响应 =================
    def on_key(self, event):
        if event.key in ['m', 'M']:
            self.mode = 'pl' if self.mode == 'p2p' else 'p2p'
            self.reset_dynamic_state()
        elif event.key in ['c', 'C'] and self.mode == 'pl' and self.state in ['adjust', 'pan_l1', 'pan_l2']:
            self.commit_measurement()
        elif event.key == 'enter':
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def on_press(self, event):
        if event.inaxes != self.ax: return

        if event.button == 3: # 右键撤销当前动态操作
            self.reset_dynamic_state()
            return

        if event.button != 1: return # 仅响应左键

        if self.mode == 'p2p':
            if self.state == 'idle':
                self.p2p_p1 = (event.xdata, event.ydata)
                self.dyn_p2p_p1.set_data([event.xdata], [event.ydata])
                self.dyn_p2p_p1.set_visible(True)
                self.dyn_p2p_line.set_visible(True)
                self.dyn_p2p_p2.set_visible(True)
                self.dyn_p2p_txt.set_visible(True)
                self.state = 'p1_selected'
                self.update_title()
            elif self.state == 'p1_selected':
                self.p2p_p2 = (event.xdata, event.ydata)
                self.commit_measurement() # P2P 直接固化

        elif self.mode == 'pl':
            if self.state == 'idle':
                self.pl_p1 = (event.xdata, event.ydata)
                self.dyn_pl_l1.set_visible(True)
                self.state = 'ref_start'
                self.update_title()
            elif self.state == 'ref_start':
                dx = event.xdata - self.pl_p1[0]
                dy = event.ydata - self.pl_p1[1]
                if np.hypot(dx, dy) < 2: return 
                self.pl_dir = (dx, dy)
                self.pl_base = ((self.pl_p1[0]+event.xdata)/2, (self.pl_p1[1]+event.ydata)/2) 
                self.dyn_pl_l2.set_visible(True)
                self.dyn_pl_perp.set_visible(True)
                self.dyn_pl_p1.set_visible(True)
                self.dyn_pl_p2.set_visible(True)
                self.dyn_pl_txt.set_visible(True)
                self.state = 'parallel'
                self.update_title()
            elif self.state == 'parallel':
                self.pl_l2_pt = (event.xdata, event.ydata)
                self.state = 'adjust'
                self.update_title()
            elif self.state == 'adjust':
                dx, dy = self.pl_dir
                A, B = -dy, dx
                norm = np.hypot(A, B)
                if norm == 0: return
                C1 = dy * self.pl_base[0] - dx * self.pl_base[1]
                C2 = dy * self.pl_l2_pt[0] - dx * self.pl_l2_pt[1]
                
                ex, ey = event.xdata, event.ydata
                d1 = abs(A*ex + B*ey + C1) / norm
                d2 = abs(A*ex + B*ey + C2) / norm

                if d1 < d2 and d1 < 40: 
                    self.state = 'pan_l1'
                    self.pan_offset = (self.pl_base[0] - ex, self.pl_base[1] - ey)
                    self.update_title()
                elif d2 <= d1 and d2 < 40: 
                    self.state = 'pan_l2'
                    self.pan_offset = (self.pl_l2_pt[0] - ex, self.pl_l2_pt[1] - ey)
                    self.update_title()

    def on_move(self, event):
        if event.inaxes != self.ax or self.bg is None: return

        if self.mode == 'p2p' and self.state == 'p1_selected':
            p1 = self.p2p_p1
            p2 = (event.xdata, event.ydata)
            dist_cm = np.hypot(p2[0]-p1[0], p2[1]-p1[1]) * self.cm_per_pixel
            
            self.dyn_p2p_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            self.dyn_p2p_p2.set_data([p2[0]], [p2[1]])
            self.dyn_p2p_txt.set_position(((p1[0]+p2[0])/2, (p1[1]+p2[1])/2))
            self.dyn_p2p_txt.set_text(f" {dist_cm:.2f} cm ")

        elif self.mode == 'pl':
            if self.state == 'ref_start':
                dx = event.xdata - self.pl_p1[0]
                dy = event.ydata - self.pl_p1[1]
                x1, y1 = self._get_extended(self.pl_p1, dx, dy)
                self.dyn_pl_l1.set_data(x1, y1)
            elif self.state in ['parallel', 'pan_l1', 'pan_l2']:
                if self.state == 'parallel':
                    self.pl_l2_pt = (event.xdata, event.ydata)
                elif self.state == 'pan_l1':
                    self.pl_base = (event.xdata + self.pan_offset[0], event.ydata + self.pan_offset[1])
                elif self.state == 'pan_l2':
                    self.pl_l2_pt = (event.xdata + self.pan_offset[0], event.ydata + self.pan_offset[1])

                p_start, p_end, dist_cm = self._calc_pl_geometry()
                dx, dy = self.pl_dir
                x1, y1 = self._get_extended(p_start, dx, dy)
                x2, y2 = self._get_extended(p_end, dx, dy)
                
                self.dyn_pl_l1.set_data(x1, y1)
                self.dyn_pl_l2.set_data(x2, y2)
                self.dyn_pl_perp.set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])
                self.dyn_pl_p1.set_data([p_start[0]], [p_start[1]])
                self.dyn_pl_p2.set_data([p_end[0]], [p_end[1]])
                
                self.dyn_pl_txt.set_position(((p_start[0]+p_end[0])/2, (p_start[1]+p_end[1])/2))
                self.dyn_pl_txt.set_text(f" D = {dist_cm:.3f} cm ")
        else:
            return

        # 高速 Blitting 局部无缝刷新引擎
        self.fig.canvas.restore_region(self.bg)
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        for a in artists:
            if a.get_visible(): self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if self.mode == 'pl' and self.state in ['pan_l1', 'pan_l2']:
            self.state = 'adjust'
            self.update_title()

# =====================================================================
# 🌟 定标专用：正弦曲线拟合函数
# =====================================================================
def sine_fit_func(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

class FCDCore:
    def __init__(self, ref_path, def_path=None, seq_dir=None, out_dir=None, crop_pixels=(0,0,0,0), 
                 water_depth=30.0, fps=30.0,low_pass_suppress=65.0, krad_factor=0.9, edge_width=10, 
                 p_low=2, p_high=98, 
                 out_hf=True, out_amp=True, out_ph=True, out_pa=True, 
                 out_disp=True, out_ndisp=True, out_sz=True, out_s3d=True, out_mom=True,
                 q_step=6, q_scale=4.0, disp_step=8, disp_scale=15.0):
        self.ref_path = ref_path
        self.def_path = def_path
        self.seq_dir = seq_dir
        self.out_dir = out_dir if out_dir else os.getcwd()
        self.crop = crop_pixels 
        
        self.H = (water_depth + 0.894 * 10.0) * 0.25
        self.fps = fps
        
        self.low_pass_suppress_r = low_pass_suppress
        self.krad_factor = krad_factor
        self.edge = edge_width
        self.p_low = p_low
        self.p_high = p_high
        
        # 🌟 新增：9 个独立的序列图窗输出开关
        self.out_hf = out_hf
        self.out_amp = out_amp
        self.out_ph = out_ph
        self.out_pa = out_pa
        self.out_disp = out_disp
        self.out_ndisp = out_ndisp
        self.out_sz = out_sz
        self.out_s3d = out_s3d
        self.out_mom = out_mom
        
        self.q_step = q_step
        self.q_scale = q_scale
        self.disp_step = disp_step     # 🌟 绑定到实例属性
        self.disp_scale = disp_scale   # 🌟 绑定到实例属性
        
        self.log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def write_log(self, prefix, content, target_dir=None):
        # 🌟 修改：支持动态传入目标保存文件夹，未传入则默认保存到基础 log 目录
        if target_dir is None:
            target_dir = self.log_dir
        os.makedirs(target_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(target_dir, f"Log_{prefix}_{timestamp}.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return log_path

    def _read_and_crop(self, path):
        img_data = np.fromfile(path, dtype=np.uint8)
        if img_data.size == 0:
            raise FileNotFoundError(f"文件不存在或为空: {path}")
        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法解码图像: {path}")
        x1, x2, y1, y2 = self.crop
        if x2 > x1 and y2 > y1:
            return img[y1:y2, x1:x2].astype(float)
        return img.astype(float)
    
    def _get_smooth_occlusion_mask(self, Iref):
        """自动抓取无纹理遮挡物，生成用于频域和梯度修复的多重掩膜"""
        # 🌟 修复之前的解包 Bug：确保缓存命中时也返回 3 个值
        if hasattr(self, '_smooth_mask') and getattr(self, '_smooth_mask').shape == Iref.shape:
            return self._smooth_mask, self._binary_water_mask, getattr(self, '_inpaint_mask')

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        local_max = cv2.dilate(Iref, kernel)
        local_min = cv2.erode(Iref, kernel)
        local_contrast = local_max - local_min

        max_contrast = np.max(local_contrast)
        _, occ_mask = cv2.threshold(local_contrast, max_contrast * 0.15, 1.0, cv2.THRESH_BINARY_INV)
        occ_mask = occ_mask.astype(np.uint8)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        occ_mask_expanded = cv2.dilate(occ_mask, dilate_kernel)

        water_mask = 1.0 - occ_mask_expanded
        smooth_mask = cv2.GaussianBlur(water_mask, (41, 41), 0)

        self._smooth_mask = smooth_mask
        self._binary_water_mask = water_mask
        
        # 🌟 新增：提取出纯粹的 8 位整型修补掩膜 (1代表需要被修补的喇叭区域)
        self._inpaint_mask = occ_mask_expanded 
        
        return self._smooth_mask, self._binary_water_mask, self._inpaint_mask

    def _subpixel_peak(self, mag, y, x):
        """
        新增辅助函数：基于局部对数抛物线拟合的亚像素寻峰
        1:1 复刻 MATLAB findpeaks2.m 中的 subpxpeak 逻辑
        """
        rows, cols = mag.shape
        if y <= 0 or y >= rows - 1 or x <= 0 or x >= cols - 1:
            return float(y), float(x)
        
        eps = 1e-10
        val = np.log(mag[y, x] + eps)
        
        # X 方向亚像素偏移
        val_l = np.log(mag[y, x - 1] + eps)
        val_r = np.log(mag[y, x + 1] + eps)
        denom_x = val_l + val_r - 2 * val
        dx = -0.5 * (val_r - val_l) / denom_x if denom_x != 0 else 0.0
        
        # Y 方向亚像素偏移
        val_u = np.log(mag[y - 1, x] + eps)
        val_d = np.log(mag[y + 1, x] + eps)
        denom_y = val_u + val_d - 2 * val
        dy = -0.5 * (val_d - val_u) / denom_y if denom_y != 0 else 0.0
        
        return y + dy, x + dx

    def _find_orth_carrier_pks(self, Iref):
        rows, cols = Iref.shape
        F = fftshift(fft2(Iref, workers=-1))
        F_mag = np.abs(F)

        cy, cx = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]

        # 保护极低频
        low_pass_suppress = 1.0 - np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * (self.low_pass_suppress_r**2)))
        F_mag_filtered = F_mag * low_pass_suppress

        # 🌟 修复：找第一载波 kr，并强制进行亚像素插值！
        y1, x1 = np.unravel_index(np.argmax(F_mag_filtered), F_mag_filtered.shape)
        y1_sub, x1_sub = self._subpixel_peak(F_mag_filtered, y1, x1)
        kr = np.array([y1_sub - cy, x1_sub - cx])

        # 正交叉乘抑制逻辑
        K_y = Y - cy
        K_x = X - cx
        K_mag = np.sqrt(K_x**2 + K_y**2)
        K_mag[K_mag == 0] = 1.0  
        kr_mag = np.sqrt(kr[0]**2 + kr[1]**2)
        
        sin_theta = np.abs(K_x * kr[0] - K_y * kr[1]) / (K_mag * kr_mag)
        F_mag_ortho = F_mag_filtered * sin_theta

        # 🌟 修复：在强制正交化后的频谱中抓取第二载波 ku，并强制进行亚像素插值！
        y2, x2 = np.unravel_index(np.argmax(F_mag_ortho), F_mag_ortho.shape)
        y2_sub, x2_sub = self._subpixel_peak(F_mag_ortho, y2, x2)
        ku = np.array([y2_sub - cy, x2_sub - cx])

        krad = (np.sqrt(np.sum((kr - ku)**2)) / 2.0) * self.krad_factor
        return kr, ku, krad
    
    def _get_bg_cache(self, shape):
        if hasattr(self, '_bg_cache') and self._bg_cache['shape'] == shape:
            return self._bg_cache

        rows, cols = shape
        X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        X_f, Y_f = X.flatten(), Y.flatten()

        # 🌟 强行降级为 2 阶基底 (6个参数)，提供绝对刚性的抛物面，无视任何高频波浪
        A = np.column_stack((
            np.ones_like(X_f), X_f, Y_f,
            X_f**2, Y_f**2, X_f*Y_f
        ))

        A_pinv = np.linalg.pinv(A)

        self._bg_cache = {
            'shape': shape,
            'A': A,
            'A_pinv': A_pinv
        }
        return self._bg_cache
    
    # def _get_bg_cache(self, shape):
    #     """生成并缓存伪逆矩阵，使得曲面拟合降维为纯乘法运算"""
    #     if hasattr(self, '_bg_cache') and self._bg_cache['shape'] == shape:
    #         return self._bg_cache

    #     rows, cols = shape
    #     X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    #     X_f, Y_f = X.flatten(), Y.flatten()

    #     A = np.column_stack((
    #         np.ones_like(X_f), X_f, Y_f,
    #         X_f**2, Y_f**2, X_f*Y_f,
    #         X_f**3, Y_f**3, (X_f**2)*Y_f, X_f*(Y_f**2)
    #     ))

    #     # 🌟 预计算 Moore-Penrose 伪逆 (耗时约0.5秒，但只执行一次)
    #     A_pinv = np.linalg.pinv(A)

    #     self._bg_cache = {
    #         'shape': shape,
    #         'A': A,
    #         'A_pinv': A_pinv
    #     }
    #     return self._bg_cache

    def _prepare_cache(self, Iref):
        if getattr(self, '_cache_valid', False) and getattr(self, '_cached_shape', None) == Iref.shape:
            return

        rows, cols = Iref.shape
        self._cached_shape = Iref.shape

        # 🌟 新增：提取遮挡物抗混叠软掩膜
        smooth_mask, _, _ = self._get_smooth_occlusion_mask(Iref)

        self.kr, self.ku, self.krad = self._find_orth_carrier_pks(Iref)
        
        # 🌟 修改：对参考图减去均值后，应用抗混叠软掩膜，彻底切断喇叭边缘的高频泄露！
        self.Fref = fft2((Iref - np.mean(Iref)) * smooth_mask, workers=-1)

        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        cy, cx = rows // 2, cols // 2

        dist2_r = (X - (cx + self.kr[1]))**2 + (Y - (cy + self.kr[0]))**2
        self.mask_r = ifftshift((dist2_r < self.krad**2).astype(float))

        dist2_u = (X - (cx + self.ku[1]))**2 + (Y - (cy + self.ku[0]))**2
        self.mask_u = ifftshift((dist2_u < self.krad**2).astype(float))

        self.cr_ref_conj = np.conj(ifft2(self.Fref * self.mask_r, workers=-1))
        self.cu_ref_conj = np.conj(ifft2(self.Fref * self.mask_u, workers=-1))

        self._cache_valid = True

    def _fcd_demodulate_correct(self, Iref, Idef):
        self._prepare_cache(Iref)
        
        # 获取掩膜
        smooth_mask, water_mask, inpaint_mask = self._get_smooth_occlusion_mask(Iref)

        Fdef = fft2((Idef - np.mean(Idef)) * smooth_mask, workers=-1)
        cr_def = ifft2(Fdef * self.mask_r, workers=-1)
        cu_def = ifft2(Fdef * self.mask_u, workers=-1)

        psi_r = cr_def * self.cr_ref_conj
        psi_u = cu_def * self.cu_ref_conj

        dphi_r = -np.angle(psi_r)
        dphi_u = -np.angle(psi_u)

        rows, cols = Iref.shape
        K_rx = 2.0 * np.pi * self.kr[1] / cols
        K_ry = 2.0 * np.pi * self.kr[0] / rows
        K_ux = 2.0 * np.pi * self.ku[1] / cols
        K_uy = 2.0 * np.pi * self.ku[0] / rows

        det = K_rx * K_uy - K_ry * K_ux
        if abs(det) < 1e-8: det = 1e-8

        d_x = (K_uy * dphi_r - K_ry * dphi_u) / det
        d_y = (-K_ux * dphi_r + K_rx * dphi_u) / det

        # 🌟 核心绝杀：不再是暴力归零，而是执行【Navier-Stokes 梯度修复】
        # 利用 cv2.inpaint 将喇叭外部的真实波浪梯度，丝滑地推演到喇叭内部！
        # 这样 Sylvester 积分器看到的就是一个完全没有物理断层的完美流场。
        d_x_32 = d_x.astype(np.float32)
        d_y_32 = d_y.astype(np.float32)

        d_x_inp = cv2.inpaint(d_x_32, inpaint_mask, 5, cv2.INPAINT_TELEA)
        d_y_inp = cv2.inpaint(d_y_32, inpaint_mask, 5, cv2.INPAINT_TELEA)

        return d_x_inp, d_y_inp

    # def _remove_background_surface(self, h, order=3):
    #     """极速曲面清洗器 (耗时从数百毫秒降低至几毫秒)"""
    #     c = self._get_bg_cache(h.shape)
        
    #     # O(N) 一步矩阵乘法直接获取 10 个曲面系数，避免每一帧都跑 lstsq 求解器
    #     C = c['A_pinv'] @ h.flatten()
    #     bg = (c['A'] @ C).reshape(h.shape)
        
    #     return h - bg
    
    def _get_sylv_cache(self, m, n):
        """生成并缓存积分器所需的所有降维、特征值与逆矩阵 (仅在第一帧耗时)"""
        if hasattr(self, '_sylv_cache') and self._sylv_cache['shape'] == (m, n):
            return self._sylv_cache

        def designgrad1D(N):
            D = np.zeros((N, N))
            idx = np.arange(1, N-1)
            D[idx, idx-1] = -0.5
            D[idx, idx+1] = 0.5
            D[0, :3] = [-1.5, 2.0, -0.5]
            D[-1, -3:] = [0.5, -2.0, 1.5]
            return D

        def housh(v):
            v = v.reshape(-1, 1)
            return np.eye(len(v)) - 2.0 * (v @ v.T) / (v.T @ v)

        Dx, Dy = designgrad1D(n), designgrad1D(m)
        vn, vm = np.ones((n, 1)), np.ones((m, 1))
        vn[0, 0] = 1.0 + np.sqrt(n)
        vm[0, 0] = 1.0 + np.sqrt(m)

        Px, Py = housh(vn), housh(vm)
        Dhx, Dhy = Dx @ Px, Dy @ Py

        A_sub = (Dhy.T @ Dhy)[1:, 1:]
        B_sub = (Dhx.T @ Dhx)[1:, 1:]

        # 🌟 极速核心 1：利用对称性一次性完成解析特征值分解，彻底干掉 solve_sylvester
        evals_A, evecs_A = np.linalg.eigh(A_sub)
        evals_B, evecs_B = np.linalg.eigh(B_sub)
        eigen_denom = evals_A[:, None] + evals_B[None, :]

        # 极速核心 2：预计算边缘权重所需矩阵的逆
        A_sub_inv = evecs_A @ np.diag(1.0 / evals_A) @ evecs_A.T
        B_sub_inv = evecs_B @ np.diag(1.0 / evals_B) @ evecs_B.T

        self._sylv_cache = {
            'shape': (m, n),
            'Px': Px, 'Py': Py, 'Dhx': Dhx, 'Dhy': Dhy,
            'evecs_A': evecs_A, 'evecs_B': evecs_B,
            'eigen_denom': eigen_denom,
            'A_sub_inv': A_sub_inv, 'B_sub_inv': B_sub_inv
        }
        return self._sylv_cache

    def _fftinvgrad(self, hx, hy):
        """
        全解析 O(N^2) 代数积分器。输出结果与原版完美一致，但速度提升几十倍！
        (为了兼容其他函数调用，名字保留为 _fftinvgrad)
        """
        m, n = hx.shape
        c = self._get_sylv_cache(m, n)

        # 提取边界参数
        C = c['Dhy'].T @ hy @ c['Px'] + c['Py'].T @ hx @ c['Dhx']
        c01, c10, C_sub = C[0, 1:], C[1:, 0], C[1:, 1:]

        # O(N^2) 极速求权重
        w01 = c['B_sub_inv'] @ c01
        w10 = c['A_sub_inv'] @ c10

        # 🌟 超级加速：在特征空间内直接用标量除法解 Sylvester 方程
        C_tilde = c['evecs_A'].T @ C_sub @ c['evecs_B']
        W_tilde = C_tilde / c['eigen_denom']
        W11 = c['evecs_A'] @ W_tilde @ c['evecs_B'].T

        # 重构并逆映射
        W = np.zeros((m, n))
        W[0, 1:], W[1:, 0], W[1:, 1:] = w01, w10, W11
        
        return c['Py'] @ W @ c['Px'].T

    def process_single_frame(self):
        Iref = self._read_and_crop(self.ref_path)
        Idef = self._read_and_crop(self.def_path)
        
        mpp = self._estimate_mm_per_pixel(Iref)
        u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
        
        e = self.edge
        u_crop = u_px[e:-e, e:-e]
        v_crop = v_px[e:-e, e:-e]
        
        u_crop -= np.mean(u_crop)
        v_crop -= np.mean(v_crop)
        
        h_px_int = self._fftinvgrad(-u_crop, -v_crop)

        h_t = h_px_int * (mpp**2) / self.H
        
        # 单帧无时序，只能进行单帧刚性 2 阶拟合用于 UI 预览
        c = self._get_bg_cache(h_t.shape)
        C_coeff = c['A_pinv'] @ h_t.flatten()
        bg = (c['A'] @ C_coeff).reshape(h_t.shape)
        h_t -= bg
        
        _, water_mask, _ = self._get_smooth_occlusion_mask(Iref)
        water_mask_crop = water_mask[e:-e, e:-e]
        
        h = h_t * water_mask_crop
        u = u_crop * mpp * water_mask_crop
        v = v_crop * mpp * water_mask_crop
        
        return h, u, v, Idef

    def _estimate_mm_per_pixel(self, Iref, G=1.5):
        kr, ku, _ = self._find_orth_carrier_pks(Iref)
        rows, cols = Iref.shape
        kr_phys = np.array([kr[1] / cols, kr[0] / rows]) * 2.0 * np.pi
        ku_phys = np.array([ku[1] / cols, ku[0] / rows]) * 2.0 * np.pi
        k0 = kr_phys + ku_phys
        kmag = np.linalg.norm(k0)
        return kmag * G / (2.0 * np.pi) if kmag > 0 else 0.12

    def _set_dynamic_ticks(self, ax, shape, mm_per_pixel):
        cm_per_pixel = mm_per_pixel / 10.0
        h_px, w_px = shape
        width_cm = w_px * cm_per_pixel
        height_cm = h_px * cm_per_pixel
        
        nice_steps = [1, 2, 5, 10, 20, 50, 100]
        target_step = max(width_cm, height_cm) / 6.0
        step_size = next((s for s in nice_steps if s >= target_step), nice_steps[-1])
        
        # 🌟 修复核心1：移除 math.ceil，确保刻度严格在图像物理尺寸内部，绝不越界撑大画框
        x_cm_main = np.arange(0, width_cm + 1e-5, step_size)
        y_cm_main = np.arange(0, height_cm + 1e-5, step_size)
        
        ax.set_xticks(x_cm_main / cm_per_pixel)
        ax.set_yticks(y_cm_main / cm_per_pixel)
        ax.set_xticklabels([f"{x:.0f}" for x in x_cm_main])
        ax.set_yticklabels([f"{y:.0f}" for y in y_cm_main])
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.tick_params(which='major', length=6, labelsize=8)
        ax.tick_params(which='minor', length=3)
        ax.set_xlabel('X (cm)', fontsize=9)
        ax.set_ylabel('Y (cm)', fontsize=9)
        # (移除了强制的 set_xlim 和 set_ylim，让 imshow 自带的边界完美贴合数据)

    def _set_colorbar_ticks(self, cbar, data, mm_per_pixel, vmin, vmax, label):
        # 1. 换算为物理真实极值
        phys_min, phys_max = vmin * mm_per_pixel, vmax * mm_per_pixel
        phys_range = max(1e-6, phys_max - phys_min)
        
        # 🌟 扩充小尺度的 nice_steps，完美应对微米级波动
        nice_steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        step_size = next((s for s in nice_steps if s >= phys_range / 5.0), nice_steps[-1])
        
        start_val = math.ceil(phys_min / step_size) * step_size
        end_val = math.floor(phys_max / step_size) * step_size
        
        # 2. 生成中间的规律刻度
        inner_ticks = np.arange(start_val, end_val + step_size * 0.1, step_size)
        
        # 3. 🌟 核心逻辑：强制首尾挂载极值，并剔除离极值太近的中间刻度 (防止数字重叠)
        threshold = step_size * 0.20 # 设定安全距离阈值为步长的 20%
        valid_inner = [t for t in inner_ticks if (t - phys_min) > threshold and (phys_max - t) > threshold]
        
        final_ticks = [phys_min] + valid_inner + [phys_max]
        
        # 设置 Colorbar 的实际刻度位置 (需要转回像素空间比率)
        cbar.set_ticks(np.array(final_ticks) / mm_per_pixel)
        
        # 4. 🌟 根据动态范围，智能分配极高精度的小数位数
        if step_size >= 1:
            fmt = "{:.0f}"
        elif step_size >= 0.1:
            fmt = "{:.1f}"
        elif step_size >= 0.01:
            fmt = "{:.2f}"
        elif step_size >= 0.001:
            fmt = "{:.3f}"
        else:
            fmt = "{:.4f}"
            
        cbar.set_ticklabels([fmt.format(x) for x in final_ticks])
        cbar.set_label(label, fontsize=10, labelpad=10)

    # 🌟 修改 1：增加 title_text 参数，在右侧图例上方补充说明绘图内容的标题
    def _draw_2d_hsv_wheel(self, ax, title_text):
        res = 150
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        
        hsv = np.zeros((res, res, 3))
        hsv[..., 0] = (phi + np.pi) / (2 * np.pi) # 角度对应色相
        hsv[..., 1] = 1.0                         # 饱和度饱和
        hsv[..., 2] = np.where(rho <= 1.0, rho, 0.0) # 半径大小对应亮度强度，圆外裁剪掉
        
        rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
        ax.imshow(rgb, extent=[-1, 1, -1, 1])
        ax.axis('off')
        
        # 在右侧图例区域添加明确的标题与物理含义说明
        ax.set_title(f"{title_text}\n\n【2D HSV 图例】\n色相(角度): 位移方向\n明度(半径): 位移强度", 
                     fontsize=10, pad=10, fontweight='bold', loc='center')

    def analyze_single_frame(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        import os, time

        if not self.ref_path or not os.path.exists(self.ref_path):
            raise ValueError("未选择有效的参考图像！")
        if not self.def_path or not os.path.exists(self.def_path):
            raise ValueError("未选择有效的形变图像！")

        ref_name = os.path.basename(self.ref_path)
        def_name = os.path.basename(self.def_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        run_dir = os.path.join(self.out_dir, "SingleFrame_Results", f"Run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)

        h_t, u_t, v_t, _ = self.process_single_frame()

        shape = h_t.shape

        # ==========================================================
        # 🌟 强化版通用绘图引擎：支持双图例与图例抗空白修复
        # ==========================================================
        def create_fig(cmap, vmin, vmax, title, label=None, is_2d=False, legend_kwargs=None):
            if is_2d:
                fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(8.5, 5), gridspec_kw={'width_ratios': [4, 1.5]}, constrained_layout=True)
                im = ax.imshow(np.zeros(shape), aspect='equal')
                ax.set_title(title, fontsize=12)
                
                max_val = legend_kwargs.get('max_val', 1.0)
                unit = legend_kwargs.get('unit', '')
                lx = legend_kwargs.get('label_x', 'X')
                ly = legend_kwargs.get('label_y', 'Y')
                is_n = legend_kwargs.get('is_norm', False)
                
                lim = max_val if not is_n else 1.0
                x = np.linspace(-lim, lim, 200)
                y = np.linspace(lim, -lim, 200) 
                Xg, Yg = np.meshgrid(x, y)
                mag = np.sqrt(Xg**2 + Yg**2)
                ang = np.mod(np.arctan2(Yg, Xg), 2*np.pi) / (2*np.pi)
                
                hsv = np.zeros((200, 200, 3))
                hsv[..., 0] = ang
                if is_n:
                    hsv[..., 1] = np.where(mag <= lim, 1.0, 0.0)
                    hsv[..., 2] = np.where(mag <= lim, 1.0, 1.0)
                    rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
                    rgb[mag > lim] = 1.0

                else:
                    hsv[..., 1] = 1.0
                    hsv[..., 2] = np.clip(mag / (max_val + 1e-10), 0, 1)
                    rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
                 
                
                ax_w.imshow(rgb, extent=[-lim, lim, -lim, lim], origin='upper')
                ax_w.set_xlabel(f"{lx} ({unit})" if unit else lx, fontsize=10)
                ax_w.set_ylabel(f"{ly} ({unit})" if unit else ly, fontsize=10)
                ax_w.set_title("正交分量映射图例", fontsize=10, weight='bold')
                ax_w.tick_params(labelsize=8)
                ax_w.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            else:
                fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
                im = ax.imshow(np.zeros(shape), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(title, fontsize=12)
                if label:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
                    self._set_colorbar_ticks(cbar, np.array([vmin, vmax]), mm_per_pixel, vmin, vmax, label=label)
            
            self._set_dynamic_ticks(ax, shape, mm_per_pixel)
            return fig, ax, im

        saved_files = []

        h_vmin, h_vmax = np.percentile(h_t, self.p_low), np.percentile(h_t, self.p_high)
        h_abs_max = max(abs(h_vmin), abs(h_vmax))
        h_vmin, h_vmax = -h_abs_max, h_abs_max
        
        # --------------------------------------------------------------------
        # 1. 保留原图：单帧瞬时水位形变
        # --------------------------------------------------------------------
        f_h, a_h, im_h = create_fig('seismic', h_vmin, h_vmax, "单帧瞬时水位形变", "水位高度 (mm)")
        im_h.set_data(h_t)
        p = os.path.join(run_dir, f'hfield_{timestamp}.jpg')
        f_h.savefig(p, dpi=150, bbox_inches='tight', pad_inches=0.02)
        saved_files.append(p)

        uv_mag = np.sqrt(u_t**2 + v_t**2)
        uv_vmax = np.percentile(uv_mag, self.p_high)
        ph_norm = np.mod(np.arctan2(v_t, u_t), 2*np.pi) / (2*np.pi)

        # --------------------------------------------------------------------
        # 🌟 2. 新增融合图：瞬时奈尔型拓扑位移场 (带双重图例)
        # --------------------------------------------------------------------
        # if getattr(self, 'out_3ddisp', True):
        if True:
            # 建立带高度 Colorbar 的底图
            f_3d, a_3d, im_3d = create_fig('seismic', h_vmin, h_vmax, "三维位移场", label="水位高度 (mm)")
            im_3d.set_data(h_t)
            
            # 🌟 修复：强行撑开顶部空间，给悬浮图例和说明文字留出充裕的安全区 (pad=35)
            a_3d.set_title("三维位移场", fontsize=12, pad=50, weight='bold')
            
            # 建立矢量格点阵列
            X_g, Y_g = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # 🌟 统一缩放逻辑：将 disp_scale 作为正向放大倍数，与动量逻辑完全一致
            disp_scale_val = getattr(self, 'disp_scale', 4.0)
            disp_width_frac = 0.1 * (disp_scale_val / 4.0)
            fixed_disp_scale = uv_vmax / (disp_width_frac + 1e-12)
            
            # 独立化：使用 disp_step 分离密度控制
            slc = slice(None, None, getattr(self, 'disp_step', 8)) 
            
            # 使用统一转换后的 fixed_disp_scale (此时 scale 越大，内部 matplotlib 的实际分母越小，箭头越长)
            q = a_3d.quiver(X_g[slc, slc], Y_g[slc, slc], u_t[slc, slc], v_t[slc, slc],
                            color='black', scale=fixed_disp_scale, scale_units='width', angles='xy', alpha=0.85)
            
            # 🌟 新增：在坐标轴左上角补充物理含义的文字说明框
            a_3d.text(0.01, 1.03, "图例说明: \n[背景] 瞬时水位高度 h (mm)\n[箭头] 面内瞬时位移场 (u, v)", 
                      transform=a_3d.transAxes, fontsize=9, va='bottom', 
                      bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

            # 🌟 修复：动态计算参考箭头并下调位置，放入撑开的顶部安全区
            exp = np.floor(np.log10(uv_vmax * 0.5))
            frac = (uv_vmax * 0.5) / 10**exp
            nice_frac = 1.0 if frac < 2 else (2.0 if frac < 5 else 5.0)
            ref_len = nice_frac * 10**exp
            qk = a_3d.quiverkey(q, X=0.88, Y=1.06, U=ref_len,
                               label=f'{ref_len:.2g} mm', labelpos='E', coordinates='axes', color='black', fontproperties={'size': 9})
            
            p_3d = os.path.join(run_dir, f'3ddisplacement_{timestamp}.jpg')
            # 略微增加一点 pad_inches 进一步防止被误裁
            f_3d.savefig(p_3d, dpi=150, bbox_inches='tight', pad_inches=0.05)
            saved_files.append(p_3d)
            plt.close(f_3d)

        # --------------------------------------------------------------------
        # 3. 保留原图：二维位移场 (修复图例后)
        # --------------------------------------------------------------------
        f_d, a_d, im_d = create_fig(None, None, None, "面内二维矢量位移场 (u, v)", is_2d=True, 
            legend_kwargs={'mode': 'cartesian', 'max_val': uv_vmax, 'unit': 'mm', 'label_x': '位移 u', 'label_y': '位移 v'})
        
        hsv_d = np.zeros((shape[0], shape[1], 3))
        hsv_d[..., 0] = ph_norm
        hsv_d[..., 1] = 1.0
        hsv_d[..., 2] = np.clip(uv_mag / (uv_vmax + 1e-10), 0, 1)
        im_d.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_d))
        p = os.path.join(run_dir, f'disp_{timestamp}.jpg')
        f_d.savefig(p, dpi=150, bbox_inches='tight', pad_inches=0.02)
        saved_files.append(p)

        # --------------------------------------------------------------------
        # 4. 保留原图：归一化位移场 (修复图例后)
        # --------------------------------------------------------------------
        f_dn, a_dn, im_dn = create_fig(None, None, None, "归一化位移场 (纯拓扑方向)", is_2d=True, 
            legend_kwargs={'mode': 'cartesian', 'max_val': 1.0, 'unit': '', 'label_x': 'u_norm', 'label_y': 'v_norm', 'is_norm': True})
        
        hsv_dn = np.zeros((shape[0], shape[1], 3))
        hsv_dn[..., 0] = ph_norm
        hsv_dn[..., 1] = 1.0
        hsv_dn[..., 2] = 1.0
        im_dn.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_dn))
        p = os.path.join(run_dir, f'norm_disp_{timestamp}.jpg')
        f_dn.savefig(p, dpi=150, bbox_inches='tight', pad_inches=0.02)
        saved_files.append(p)

        plt.close('all')

        # --------------------------------------------------------------------
        # 5. 保留原有功能：输出三大物理场绝对矩阵 CSV
        # --------------------------------------------------------------------
        np.savetxt(os.path.join(run_dir, f'hfield_matrix_{timestamp}.csv'), h_t, delimiter=',', fmt='%.5f')
        np.savetxt(os.path.join(run_dir, f'disp_u_matrix_{timestamp}.csv'), u_t, delimiter=',', fmt='%.5f')
        np.savetxt(os.path.join(run_dir, f'disp_v_matrix_{timestamp}.csv'), v_t, delimiter=',', fmt='%.5f')

        log_c = (f"===== 单帧静力学与拓扑分析完成 =====\n"
                 f"静态参考图: {ref_name}\n"
                 f"动态形变图: {def_name}\n"
                 f"打包输出目录: {run_dir}\n"
                 f"已成功导出 {len(saved_files)} 张带有标准图例的物理与拓扑图像。\n"
                 f"已成功导出 3 份原始物理矩阵 CSV 数据 (h, u, v)。")
        
        return self.write_log("SingleFrame", log_c, target_dir=run_dir)

    def find_pixels(self):
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        img = self._read_and_crop(self.ref_path)
        img_height, img_width = img.shape[:2]

        fig, ax = plt.subplots(num='点击图片获取坐标')
        ax.imshow(img, cmap='gray')
        ax.set_title('【单击】定起点 → 【移动】拉出红色虚线正方形 → 【再次单击】确认截取', fontsize=11)
        plt.axis('on')

        # 状态字典加入 'bg' 用于存储背景图像缓存（Blitting 提速关键）
        state = {'start': None, 'rect': None, 'bg': None}
        points = []

        def get_square_coords(start_x, start_y, curr_x, curr_y):
            """计算以起点为基准的最大正方形"""
            dx = curr_x - start_x
            dy = curr_y - start_y
            
            # 🌟 修复 1：改为 max，使正方形的边始终“贴”在鼠标移动较远的那一维上
            side = max(abs(dx), abs(dy))
            sign_x = 1 if dx > 0 else -1
            sign_y = 1 if dy > 0 else -1
            return side, sign_x, sign_y

        def onmove(event):
            # 如果没定起点，或背景未缓存，或鼠标移出坐标轴，则不处理
            if not state['start'] or not state['bg'] or event.inaxes != ax:
                return
            
            side, sign_x, sign_y = get_square_coords(state['start'][0], state['start'][1], event.xdata, event.ydata)
            state['rect'].set_width(side * sign_x)
            state['rect'].set_height(side * sign_y)
            
            # 🌟 修复 2：Blit 局部高速重绘技术。恢复背景 -> 仅画红框 -> 刷新局部屏幕
            fig.canvas.restore_region(state['bg'])
            ax.draw_artist(state['rect'])
            fig.canvas.blit(ax.bbox)

        def onclick(event):
            if event.inaxes != ax or event.button != 1:
                return

            if not state['start']:
                # 第一次点击：确定起点
                x, y = event.xdata, event.ydata
                state['start'] = (x, y)
                
                # 生成红色虚线框初始对象
                state['rect'] = patches.Rectangle((x, y), 0, 0, linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(state['rect'])
                ax.plot(x, y, 'r+', markersize=10, linewidth=1.5)
                
                # 画完初始十字准星后，截取当前纯净的背景（用于后续高速滑动刷新）
                fig.canvas.draw()
                state['bg'] = fig.canvas.copy_from_bbox(ax.bbox)
            else:
                # 第二次点击：固定正方形并结束
                side, sign_x, sign_y = get_square_coords(state['start'][0], state['start'][1], event.xdata, event.ydata)
                
                # 防止原地误触双击导致崩溃
                if side < 5: return

                end_x = state['start'][0] + side * sign_x
                end_y = state['start'][1] + side * sign_y

                # 约束坐标不要超出图像物理边界
                start_x_c = max(0, min(img_width-1, state['start'][0]))
                start_y_c = max(0, min(img_height-1, state['start'][1]))
                end_x_c = max(0, min(img_width-1, end_x))
                end_y_c = max(0, min(img_height-1, end_y))

                points.append((int(round(start_x_c)), int(round(start_y_c))))
                points.append((int(round(end_x_c)), int(round(end_y_c))))
                plt.close(fig)

        def onkey(event):
            if event.key in ['enter', 'escape']:
                plt.close(fig)

        # 绑定事件
        fig.canvas.mpl_connect('motion_notify_event', onmove)
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()

        # 原代码的结尾部分：
        matplotlib.use('Agg', force=True)
        
        # 🌟 修改：提取图片名，并将日志重定向到 FindPixels_Results 文件夹
        ref_name = os.path.basename(self.ref_path) if self.ref_path else "未知"
        find_dir = os.path.join(self.out_dir, "FindPixels_Results")
        log_c = f"操作: 获取像素坐标\n静态参考图: {ref_name}\n采集点数: {len(points)}\n绝对坐标: {points}"
        
        return points, self.write_log("FindPixels", log_c, target_dir=find_dir)

    def measure_distance(self):
        import matplotlib
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        import os, time

        if not self.ref_path or not self.def_path:
            raise ValueError("请先配置静态参考图和形变图路径！")

        # 1. 基础读取与像素刻度计算
        Iref = self._read_and_crop(self.ref_path)
        Idef = self._read_and_crop(self.def_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        cm_per_pixel = mm_per_pixel / 10.0

        # 2. 解调出水位场 (高度图) 用于测量背景
        u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
        # 🌟 终极修复：先切除 FCD 污染边缘，再对纯净场逆积分
        e = self.edge
        u_crop = u_px[e:-e, e:-e]
        v_crop = v_px[e:-e, e:-e]
        
        h_px_int = self._fftinvgrad(-u_crop, -v_crop)
        h_t = h_px_int * (mm_per_pixel**2) / self.H
        # h = self._remove_background_surface(h_t, order=3)
        shape = h_t.shape

        # 3. 构建交互画板
        fig, ax = plt.subplots(figsize=(8, 6), num='FCD 交互式距离测量')
        h_vmin, h_vmax = np.percentile(h_t, self.p_low), np.percentile(h_t, self.p_high)
        im = ax.imshow(h_t, cmap='jet', vmin=h_vmin, vmax=h_vmax, aspect='equal')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
        self._set_colorbar_ticks(cbar, np.array([h_vmin, h_vmax]), mm_per_pixel, h_vmin, h_vmax, label='水位高度 (mm)')
        self._set_dynamic_ticks(ax, shape, mm_per_pixel)

        # 4. 挂载交互测距工具并阻塞等待用户操作
        measurer = InteractiveMeasurer(ax, fig, cm_per_pixel)
        plt.show(block=True)
        matplotlib.use('Agg', force=True) 

        ref_name = os.path.basename(self.ref_path)
        def_name = os.path.basename(self.def_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.out_dir, "Measure_Results", f"Run_{timestamp}")

        # 5. 如果用户进行了测量，则保存结果
        if len(measurer.measurements) > 0:
            os.makedirs(run_dir, exist_ok=True)
            base_name = os.path.splitext(def_name)[0]
            
            ax.set_title("") 
            img_path = os.path.join(run_dir, f"{base_name}_height_measured_{timestamp}.png")
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            
            log_path = os.path.join(run_dir, f"Log_measure_cm_{base_name}_{timestamp}.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("===== 处理日志 =====\n")
                f.write(f"处理时间: {timestamp}\n")
                f.write(f"静态参考图: {ref_name} ({self.ref_path})\n")
                f.write(f"动态形变图: {def_name} ({self.def_path})\n")
                f.write(f"裁剪参数: {self.crop}\n\n")
                
                f.write("===== 距离测量结果 =====\n")
                f.write(f"物理单位: cm (每像素 = {cm_per_pixel:.4f} cm)\n")
                f.write("测量时间\t序号\t点1坐标(px)\t点2坐标(px)\t距离(cm)\n")
                
                for m in measurer.measurements:
                    p1, p2 = m['p1'], m['p2']
                    f.write(f"{m['time']}\t{m['id']}\t[{p1[0]:.1f}, {p1[1]:.1f}]\t[{p2[0]:.1f}, {p2[1]:.1f}]\t{m['dist_cm']:.2f}\n")

            summary = [m['dist_cm'] for m in measurer.measurements]
            log_str = f"交互测距完成！\n静态参考图: {ref_name}\n动态形变图: {def_name}\n共测量 {len(summary)} 组距离。\n结果 (cm): {', '.join([f'{d:.2f}' for d in summary])}\n截图与日志已打包保存至输出目录。"
            return summary[-1] if summary else 0, self.write_log("Measure", log_str, target_dir=run_dir)
        else:
            log_str = f"交互测距已取消或未提取任何点。\n静态参考图: {ref_name}\n动态形变图: {def_name}"
            return 0, self.write_log("Measure_Cancel", log_str)

    def calculate_q_value(self):
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector
        h, u, v, _ = self.process_single_frame()
        R_mag = np.sqrt(u**2 + v**2 + h**2 + 1e-10)
        un, vn, hn = u/R_mag, v/R_mag, h/R_mag
        du_dy, du_dx = np.gradient(un)
        dv_dy, dv_dx = np.gradient(vn)
        dh_dy, dh_dx = np.gradient(hn)
        integrand = np.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                dR_dx = np.array([du_dx[i,j], dv_dx[i,j], dh_dx[i,j]])
                dR_dy = np.array([du_dy[i,j], dv_dy[i,j], dh_dy[i,j]])
                integrand[i,j] = np.dot(np.array([un[i,j], vn[i,j], hn[i,j]]), np.cross(dR_dx, dR_dy))
        fig, ax = plt.subplots(num="框选区域计算 Q 值")
        ax.imshow(h, cmap='jet')
        poly_pts = []
        def onselect(verts):
            poly_pts.clear()
            poly_pts.extend(verts)
        selector = PolygonSelector(ax, onselect)
        plt.show()
        matplotlib.use('Agg', force=True)
        if len(poly_pts) < 3: return None, None
        x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        mask = Path(poly_pts).contains_points(np.vstack((x.flatten(), y.flatten())).T).reshape(u.shape)
        Q = np.sum(integrand[mask]) / (4 * np.pi)
        return Q, self.write_log("Qvalue", f"Q值: {Q:.4f}")

    def _process_frame_worker(self, def_path, Iref, mm_per_pixel):
        """线程池专属 Worker：负责极速解调单张物理场"""
        Idef = self._read_and_crop(def_path)
        u_px, v_px = self._fcd_demodulate_correct(Iref, Idef)
        
        e = self.edge
        u_crop = u_px[e:-e, e:-e]
        v_crop = v_px[e:-e, e:-e]
        
        # 🌟 极简去均值：强制消灭相机常数级平移引发的贯穿全屏的一阶大斜坡
        u_crop -= np.mean(u_crop)
        v_crop -= np.mean(v_crop)
        
        h_px_int = self._fftinvgrad(-u_crop, -v_crop)
        
        h_t = h_px_int * (mm_per_pixel**2) / self.H
        
        _, water_mask, _ = self._get_smooth_occlusion_mask(Iref)
        water_mask_crop = water_mask[e:-e, e:-e]

        h = h_t * water_mask_crop
        u_t = u_crop * mm_per_pixel * water_mask_crop
        v_t = v_crop * mm_per_pixel * water_mask_crop
        
        return h, u_t, v_t

    def process_sequence(self):
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        if not self.seq_dir or not os.path.exists(self.seq_dir):
            raise ValueError("图片序列目录无效或不存在！")

        files = sorted([f for f in os.listdir(self.seq_dir) if f.endswith(('.bmp', '.tiff', '.png', '.jpg'))])
        if not files: raise ValueError("没有找到有效图像帧")
        
        seq_out_dir = os.path.join(self.out_dir, os.path.basename(self.seq_dir) + "_results")
        os.makedirs(seq_out_dir, exist_ok=True)
        
        # 动态组装子输出目录
        subdirs = []
        if self.out_hf: subdirs.append('hfield')
        if self.out_amp: subdirs.append('amplitude')
        if self.out_ph: subdirs.append('phase')
        if self.out_pa: subdirs.append('phaseamp')
        if self.out_disp: subdirs.append('displacement')
        if self.out_ndisp: subdirs.append('norm_disp')
        if self.out_sz: subdirs.append('sz')
        if self.out_s3d: subdirs.append('s2d')
        if self.out_mom: subdirs.append('momentum')
        if getattr(self, 'out_3ddisp', True): subdirs.append('3d_displacement')
        if getattr(self, 'out_3dspin', True): subdirs.append('s3d')
        for d in subdirs: os.makedirs(os.path.join(seq_out_dir, d), exist_ok=True)

        Iref = self._read_and_crop(self.ref_path)
        mm_per_pixel = self._estimate_mm_per_pixel(Iref)
        frames = len(files) 
        
        print("正在准备计算...")
        dummy_u = np.zeros((Iref.shape[0]-2*self.edge, Iref.shape[1]-2*self.edge))
        self._fftinvgrad(dummy_u, dummy_u)
        
        h_list, u_list, v_list = [None] * frames, [None] * frames, [None] * frames
        import concurrent.futures
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        print(f"启动CPU多核并发 (并发数: {max_workers})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_frame_worker, os.path.join(self.seq_dir, files[i]), Iref, mm_per_pixel): i
                for i in range(frames)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                h, u, v = future.result()
                h_list[idx] = h; u_list[idx] = u; v_list[idx] = v

        import gc
        h_stack = np.array(h_list, dtype=np.float32)
        u_stack = np.array(u_list, dtype=np.float32)
        v_stack = np.array(v_list, dtype=np.float32)
        shape = h_stack[0].shape
        del h_list, u_list, v_list; gc.collect()

        fps = getattr(self, 'fps', 30.0) 
        
        # ==========================================================
        # 🌟 终极科学清理：时均场静态本底统一扣除法
        # ==========================================================
        print("🌊 正在提取并扣除全局统一静态积分本底大碗...")
        # 1. 获取全局时间平均高度场 (残留的非整数周期水波被极大削弱)
        h_mean_time = np.mean(h_stack, axis=0)
        
        # 2. 从时均场中拟合出绝对静止的 2 阶背景大碗 
        # (刚性抛物面完全无视高频残留水波，完美解决 DC 泄露悖论)
        c = self._get_bg_cache(h_mean_time.shape)
        C = c['A_pinv'] @ h_mean_time.flatten()
        bg_static = (c['A'] @ C).reshape(h_mean_time.shape)
        
        # 3. 对所有帧统一扣除这个绝对静止的本底！
        h_stack -= bg_static
        # ==========================================================

        # 提取时域标准差静态包络
        amp_map = np.std(h_stack, axis=0) * np.sqrt(2.0)

        h_ana = np.conj(hilbert(h_stack, axis=0)).astype(np.complex64)
        del h_stack
        
        u_ana = np.conj(hilbert(u_stack, axis=0)).astype(np.complex64)
        del u_stack
        
        v_ana = np.conj(hilbert(v_stack, axis=0)).astype(np.complex64)
        del v_stack
        gc.collect()

        # pad_len = frames // 2
        # pad_cfg = ((pad_len, pad_len), (0, 0), (0, 0))
        
        # # 使用 reflect 配合 odd 模式
        # h_pad = np.pad(h_stack, pad_cfg, mode='reflect', reflect_type='odd')
        # del h_stack
        # h_ana = np.conj(hilbert(h_pad, axis=0)[pad_len : pad_len + frames]).astype(np.complex64)
        # del h_pad
        # gc.collect()
        
        # u_pad = np.pad(u_stack, pad_cfg, mode='reflect', reflect_type='odd')
        # del u_stack
        # u_ana = np.conj(hilbert(u_pad, axis=0)[pad_len : pad_len + frames]).astype(np.complex64)
        # del u_pad
        # gc.collect()
        
        # v_pad = np.pad(v_stack, pad_cfg, mode='reflect', reflect_type='odd')
        # del v_stack
        # v_ana = np.conj(hilbert(v_pad, axis=0)[pad_len : pad_len + frames]).astype(np.complex64)
        # del v_pad
        # gc.collect()

        phase_w = np.mod(np.angle(h_ana), 2*np.pi)
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        h_ana /= 1000.0; u_ana /= 1000.0; v_ana /= 1000.0
        u_vel_m = np.gradient(u_ana, axis=0) / dt
        v_vel_m = np.gradient(v_ana, axis=0) / dt
        h_vel_m = np.gradient(h_ana, axis=0) / dt
        
        calc_spin = self.out_sz or self.out_s3d or getattr(self, 'out_3dspin', True)
        if calc_spin:
            sx = -np.real(np.conj(v_ana) * h_vel_m - np.conj(h_ana) * v_vel_m) * 1e6
            sy =  np.real(np.conj(h_ana) * u_vel_m - np.conj(u_ana) * h_vel_m) * 1e6
            sz = -np.real(np.conj(u_ana) * v_vel_m - np.conj(v_ana) * u_vel_m) * 1e6
            
        safe_margin = max(3, frames // 10)
        safe_slice = slice(safe_margin, frames - safe_margin)

        h_real_mm = np.real(h_ana) * 1000.0
        h_vmin, h_vmax = np.percentile(h_real_mm[safe_slice], self.p_low), np.percentile(h_real_mm[safe_slice], self.p_high)
        h_abs_max = max(abs(h_vmin), abs(h_vmax))
        h_vmin, h_vmax = -h_abs_max, h_abs_max
        del h_real_mm; gc.collect()

        amp_vmin, amp_vmax = np.percentile(amp_map, self.p_low), np.percentile(amp_map, self.p_high)
        if calc_spin:
            sz_vmin, sz_vmax = np.percentile(sz[safe_slice], self.p_low), np.percentile(sz[safe_slice], self.p_high)

        if self.out_disp or self.out_ndisp or getattr(self, 'out_3ddisp', True):
            uv_mag_all = np.sqrt(np.real(u_ana)**2 + np.real(v_ana)**2) * 1000.0
            uv_vmax = np.percentile(uv_mag_all[safe_slice], self.p_high)
            del uv_mag_all; gc.collect()
            
        if self.out_s3d:
            sxy_mag_all = np.sqrt(sx**2 + sy**2)
            sxy_vmax = np.percentile(sxy_mag_all[safe_slice], self.p_high)
            del sxy_mag_all; gc.collect()

        if self.out_mom:
            mpp_m = mm_per_pixel / 1000.0
            du_dy_m, du_dx_m = np.gradient(u_ana, axis=(1, 2))
            du_dy_m /= mpp_m; du_dx_m /= mpp_m
            dv_dy_m, dv_dx_m = np.gradient(v_ana, axis=(1, 2))
            dv_dy_m /= mpp_m; dv_dx_m /= mpp_m
            dw_dy_m, dw_dx_m = np.gradient(h_ana, axis=(1, 2))
            dw_dy_m /= mpp_m; dw_dx_m /= mpp_m
            Px_all = -np.real(np.conj(u_vel_m)*du_dx_m + np.conj(v_vel_m)*dv_dx_m + np.conj(h_vel_m)*dw_dx_m)
            Py_all = -np.real(np.conj(u_vel_m)*du_dy_m + np.conj(v_vel_m)*dv_dy_m + np.conj(h_vel_m)*dw_dy_m)
            Px_all *= 1000.0; Py_all *= 1000.0
            del du_dy_m, du_dx_m, dv_dy_m, dv_dx_m, dw_dy_m, dw_dx_m; gc.collect()
            global_max_p = np.percentile(np.sqrt(Px_all**2 + Py_all**2)[safe_slice], 99.5)
            if global_max_p == 0 or np.isnan(global_max_p): global_max_p = 1e-12
            exp = np.floor(np.log10(global_max_p * 0.4))
            frac = (global_max_p * 0.4) / 10**exp
            nice_frac = 1.0 if frac < 2 else (2.0 if frac < 5 else 5.0)
            ref_len = nice_frac * 10**exp
            arrow_width_fraction = 0.1 * (self.q_scale / 4.0)
            fixed_scale = global_max_p / (arrow_width_fraction + 1e-12)

        def create_fig(cmap, vmin, vmax, title, label=None, is_2d=False, legend_kwargs=None):
            if is_2d:
                fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(8.5, 5), gridspec_kw={'width_ratios': [4, 1.5]}, constrained_layout=True)
                im = ax.imshow(np.zeros(shape), aspect='equal')
                ax.set_title(title, fontsize=11, weight='bold')
                mode = legend_kwargs.get('mode', 'cartesian')
                max_val = legend_kwargs.get('max_val', 1.0)
                unit = legend_kwargs.get('unit', '')

                if mode == 'cartesian':
                    lx, ly = legend_kwargs.get('label_x', 'X'), legend_kwargs.get('label_y', 'Y')
                    is_n = legend_kwargs.get('is_norm', False)
                    lim = max_val if not is_n else 1.0
                    x, y = np.linspace(-lim, lim, 200), np.linspace(lim, -lim, 200)
                    Xg, Yg = np.meshgrid(x, y)
                    mag = np.sqrt(Xg**2 + Yg**2)
                    ang = np.mod(np.arctan2(Yg, Xg), 2*np.pi) / (2*np.pi)
                    hsv = np.zeros((200, 200, 3))
                    hsv[..., 0] = ang
                    if is_n:
                        hsv[..., 1] = np.where(mag <= lim, 1.0, 0.0); hsv[..., 2] = np.where(mag <= lim, 1.0, 1.0)
                        rgb = plt.matplotlib.colors.hsv_to_rgb(hsv); rgb[mag > lim] = 1.0
                    else:
                        hsv[..., 1] = 1.0; hsv[..., 2] = np.clip(mag / (max_val + 1e-10), 0, 1)
                        rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)
                    ax_w.imshow(rgb, extent=[-lim, lim, -lim, lim])
                    ax_w.set_xlabel(f"{lx} ({unit})" if unit else lx, fontsize=9)
                    ax_w.set_ylabel(f"{ly} ({unit})" if unit else ly, fontsize=9)
                    ax_w.set_title("正交分量映射图例", fontsize=9, weight='bold')

                elif mode == 'polar_rect':
                    lx, ly = legend_kwargs.get('label_x', '相位 (rad)'), legend_kwargs.get('label_y', '幅值 (mm)')
                    x, y = np.linspace(-np.pi, np.pi, 200), np.linspace(max_val, 0, 200)
                    Xg, Yg = np.meshgrid(x, y)
                    hsv = np.zeros((200, 200, 3))
                    hsv[..., 0] = (Xg + np.pi) / (2 * np.pi); hsv[..., 1] = 1.0; hsv[..., 2] = np.clip(Yg / (max_val + 1e-10), 0, 1)
                    ax_w.imshow(plt.matplotlib.colors.hsv_to_rgb(hsv), extent=[-np.pi, np.pi, 0, max_val], aspect='auto')
                    ax_w.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]); ax_w.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
                    ax_w.set_xlabel(lx, fontsize=9); ax_w.set_ylabel(ly, fontsize=9)
                    ax_w.set_title("相幅映射图例", fontsize=9, weight='bold')

                elif mode == 'spin_topology':
                    x, y = np.linspace(-np.pi, np.pi, 200), np.linspace(1.0, -1.0, 200)
                    Xg, Yg = np.meshgrid(x, y)
                    hsv = np.zeros((200, 200, 3))
                    hsv[..., 0] = (Xg + np.pi) / (2 * np.pi)
                    hsv[..., 1] = np.where(Yg > 0, 1.0 - Yg, 1.0)
                    hsv[..., 2] = np.where(Yg > 0, 1.0, 1.0 + Yg)
                    ax_w.imshow(plt.matplotlib.colors.hsv_to_rgb(hsv), extent=[-np.pi, np.pi, -1.0, 1.0], aspect='auto')
                    ax_w.set_xticks([-np.pi, 0, np.pi]); ax_w.set_xticklabels(['-π', '0', 'π'])
                    ax_w.set_yticks([-1, 0, 1]); ax_w.set_yticklabels(['-1', '0', '+1'])
                    ax_w.set_xlabel(r"横向自旋角 $\phi$ (rad)", fontsize=9); ax_w.set_ylabel("纵向极化 $S_z$ (归一化)", fontsize=9)
                    ax_w.set_title("自旋映射图例", fontsize=9, weight='bold')

                ax_w.tick_params(labelsize=8); ax_w.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            else:
                fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
                im = ax.imshow(np.zeros(shape), cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(title, fontsize=12)
                if label:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
                    if title == "相位角":
                        cbar.set_ticks([0, np.pi, 2*np.pi]); cbar.set_ticklabels(['0', 'π', '2π'])
                    else:
                        self._set_colorbar_ticks(cbar, np.array([vmin, vmax]), mm_per_pixel, vmin, vmax, label=label)
            self._set_dynamic_ticks(ax, shape, mm_per_pixel)
            return fig, ax, im

        # 初始化标准绘图窗口
        if self.out_hf: f_h, a_h, im_h = create_fig('seismic', h_vmin, h_vmax, "水位形变", "水位形变 (mm)")
        if self.out_ph: f_p, a_p, im_p = create_fig('hsv', 0, 2*np.pi, "相位角", "相位 (rad)")
        if self.out_sz: f_sz, a_sz, im_sz = create_fig('viridis', sz_vmin, sz_vmax, "Z向自旋角动量", "自旋密度 ($mm^2/s$)")
        if self.out_pa: f_pa, a_pa, im_pa = create_fig(None, None, None, "相位振幅复合场", is_2d=True, legend_kwargs={'mode': 'polar_rect', 'max_val': amp_vmax})
        if self.out_disp: f_d, a_d, im_d = create_fig(None, None, None, "面内二维矢量位移场 (u, v)", is_2d=True, legend_kwargs={'mode': 'cartesian', 'max_val': uv_vmax, 'unit': 'mm', 'label_x': '位移 u', 'label_y': '位移 v'})
        if self.out_ndisp: f_dn, a_dn, im_dn = create_fig(None, None, None, "归一化位移场 (纯拓扑方向)", is_2d=True, legend_kwargs={'mode': 'cartesian', 'max_val': 1.0, 'unit': '', 'label_x': 'u_norm', 'label_y': 'v_norm', 'is_norm': True})
        if self.out_s3d: f_s3, a_s3, im_s3 = create_fig(None, None, None, "横向自旋角动量场 (Sx, Sy)", is_2d=True, legend_kwargs={'mode': 'cartesian', 'max_val': sxy_vmax, 'unit': '$mm^2/s$', 'label_x': 'Sx', 'label_y': 'Sy'})
        
        # 🌟 修复问题 3：动量流场窗口初始化，注入大 pad 并添加左上角物理文本说明框
        if self.out_mom:
            f_m, a_m, im_m = create_fig(None, None, None, "动量密度流场", is_2d=True, legend_kwargs={'mode': 'polar_rect', 'max_val': amp_vmax})
            a_m.set_title("动量密度流场", fontsize=11, pad=50, weight='bold')
            a_m.text(0.02, 1.04, "图例说明: \n[背景] 彩色相幅复合场 (亮度=振幅, 色相=相位)\n[箭头] 动量流 / 斯托克斯漂移速度", 
                     transform=a_m.transAxes, fontsize=8, va='bottom', bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.2'))
        
        X_grid, Y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # 🌟 独立化：使用 disp_step 代替共用的 q_step
        slc_3d = slice(None, None, getattr(self, 'disp_step', 8))
        # 🌟 统一缩放逻辑：将 disp_scale 作为正向放大倍数，与动量逻辑完全一致
        disp_scale_val = getattr(self, 'disp_scale', 4.0)
        disp_width_frac = 0.1 * (disp_scale_val / 4.0)
        fixed_disp_scale = uv_vmax / (disp_width_frac + 1e-12)

        # 🌟 修复问题 4：Neel 位移场初始化，指定 label="水位高度 (mm)" 强制生成右侧高度图例，并增加物理说明框与顶 pad
        if getattr(self, 'out_3ddisp', True):
            f_3dd, a_3dd, im_3dd = create_fig('seismic', h_vmin, h_vmax, "三维位移场", label="水位高度 (mm)")
            a_3dd.set_title("三维位移场", fontsize=11, pad=50, weight='bold')
            a_3dd.text(0.02, 1.04, "图例说明: \n[背景] 瞬时水位高度 h (mm)\n[箭头] 面内瞬时位移场 (u, v)", 
                       transform=a_3dd.transAxes, fontsize=8, va='bottom', bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.2'))
            
            # 使用统一转换后的 fixed_disp_scale
            q_3dd = a_3dd.quiver(X_grid[slc_3d, slc_3d], Y_grid[slc_3d, slc_3d], 
                                 np.zeros_like(X_grid[slc_3d, slc_3d]), np.zeros_like(Y_grid[slc_3d, slc_3d]),
                                 color='black', scale=fixed_disp_scale, scale_units='width', angles='xy', alpha=0.8)
            
            # 🌟 修复问题 4：添加固定的面内矢量参考箭头标尺 (QuiverKey)
            exp_3d = np.floor(np.log10(uv_vmax * 0.5))
            frac_3d = (uv_vmax * 0.5) / 10**exp_3d
            nice_frac_3d = 1.0 if frac_3d < 2 else (2.0 if frac_3d < 5 else 5.0)
            ref_len_3d = nice_frac_3d * 10**exp_3d
            qk_3dd = a_3dd.quiverkey(q_3dd, X=0.88, Y=1.06, U=ref_len_3d,
                                    label=f'{ref_len_3d:.2g} mm', labelpos='E', coordinates='axes', color='black', fontproperties={'size': 8})

        # 🌟 修复问题 2：全分量自旋场初始化，注入大 pad 彻底消除说明文字与主标题重合
        if getattr(self, 'out_3dspin', True):
            f_3ds, a_3ds, im_3ds = create_fig(None, None, None, "全分量自旋场", is_2d=True, legend_kwargs={'mode': 'spin_topology'})
            a_3ds.set_title("全分量自旋场", fontsize=11, pad=50, weight='bold')
            a_3ds.text(0.02, 1.04, "拓扑映射说明:\n色相(Hue) -> 横向自旋角\n明暗(V-S) -> Z向偏振极化", 
                       transform=a_3ds.transAxes, fontsize=8, va='bottom', bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.2'))

        # 🌟 修复问题 1：在批处理正式循环前，计算单图全量振幅包络并独立安全落盘
        if self.out_amp:
            f_a, a_a, im_a = create_fig('hot', amp_vmin, amp_vmax, "全局水波振幅", "振幅 (mm)")
            im_a.set_data(amp_map)
            f_a.savefig(os.path.join(seq_out_dir, 'amplitude', 'Global_Amplitude_Envelope.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            plt.close(f_a)

        # ==========================================================
        # 批量时间轴高速循环渲染器
        # ==========================================================
        for t in range(frames):
            tag = f"{t:03d}"
            
            if self.out_hf:
                im_h.set_data(np.real(h_ana[t]) * 1000.0)
                f_h.savefig(os.path.join(seq_out_dir, 'hfield', f'hfield_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            if getattr(self, 'out_3ddisp', True):
                im_3dd.set_data(np.real(h_ana[t]) * 1000.0)
                u_fr, v_fr = np.real(u_ana[t]) * 1000.0, np.real(v_ana[t]) * 1000.0
                q_3dd.set_UVC(u_fr[slc_3d, slc_3d], v_fr[slc_3d, slc_3d])
                f_3dd.savefig(os.path.join(seq_out_dir, '3d_displacement', f'3d_disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.05)

            if getattr(self, 'out_3dspin', True):
                spin_hsv = np.zeros((shape[0], shape[1], 3))
                spin_hsv[..., 0] = (np.arctan2(sy[t], sx[t]) + np.pi) / (2 * np.pi)
                sz_max_val = max(abs(sz_vmin), abs(sz_vmax)) + 1e-12
                sz_norm = np.clip(sz[t] / sz_max_val, -1.0, 1.0)
                spin_hsv[..., 1] = np.where(sz_norm > 0, 1.0 - sz_norm, 1.0)  
                spin_hsv[..., 2] = np.where(sz_norm > 0, 1.0, 1.0 + sz_norm)  
                im_3ds.set_data(plt.matplotlib.colors.hsv_to_rgb(spin_hsv))
                f_3ds.savefig(os.path.join(seq_out_dir, 's3d', f'full_spin_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.05)

            if self.out_ph:
                im_p.set_data(phase_w[t])
                f_p.savefig(os.path.join(seq_out_dir, 'phase', f'phase_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            if self.out_pa:
                pa_hsv = np.zeros((shape[0], shape[1], 3))
                pa_hsv[..., 0] = (np.angle(h_ana[t]) + np.pi) / (2 * np.pi); pa_hsv[..., 1] = 1.0; pa_hsv[..., 2] = np.clip(amp_map / (amp_vmax + 1e-10), 0, 1)
                im_pa.set_data(plt.matplotlib.colors.hsv_to_rgb(pa_hsv))
                f_pa.savefig(os.path.join(seq_out_dir, 'phaseamp', f'phaseamp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            if self.out_disp or self.out_ndisp:
                u_r, v_r = np.real(u_ana[t]) * 1000.0, np.real(v_ana[t]) * 1000.0
                ph_norm = np.mod(np.arctan2(v_r, u_r), 2*np.pi) / (2*np.pi)
                if self.out_disp:
                    hsv_d = np.zeros((shape[0], shape[1], 3)); hsv_d[..., 0] = ph_norm; hsv_d[..., 1] = 1.0; hsv_d[..., 2] = np.clip(np.sqrt(u_r**2 + v_r**2) / (uv_vmax + 1e-10), 0, 1)
                    im_d.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_d))
                    f_d.savefig(os.path.join(seq_out_dir, 'displacement', f'disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
                if self.out_ndisp:
                    hsv_dn = np.zeros((shape[0], shape[1], 3)); hsv_dn[..., 0] = ph_norm; hsv_dn[..., 1] = 1.0; hsv_dn[..., 2] = 1.0
                    im_dn.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_dn))
                    f_dn.savefig(os.path.join(seq_out_dir, 'norm_disp', f'norm_disp_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            if self.out_sz:
                im_sz.set_data(sz[t])
                f_sz.savefig(os.path.join(seq_out_dir, 'sz', f'sz_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            if self.out_s3d:
                hsv_s3 = np.zeros((shape[0], shape[1], 3)); hsv_s3[..., 0] = np.mod(np.arctan2(sy[t], sx[t]), 2*np.pi) / (2*np.pi); hsv_s3[..., 1] = 1.0; hsv_s3[..., 2] = np.clip(np.sqrt(sx[t]**2 + sy[t]**2) / (sxy_vmax + 1e-10), 0, 1)
                im_s3.set_data(plt.matplotlib.colors.hsv_to_rgb(hsv_s3))
                f_s3.savefig(os.path.join(seq_out_dir, 's2d', f's2d_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.02)
            
            # 🌟 修复问题 3：动量流场微调 quiverkey 参考箭头坐标，放入撑开的顶部安全区
            if self.out_mom:
                pa_hsv = np.zeros((shape[0], shape[1], 3)); pa_hsv[..., 0] = (np.angle(h_ana[t]) + np.pi) / (2 * np.pi); pa_hsv[..., 1] = 1.0; pa_hsv[..., 2] = np.clip(amp_map / (amp_vmax + 1e-10), 0, 1)
                im_m.set_data(plt.matplotlib.colors.hsv_to_rgb(pa_hsv))
                a_m.set_xlim(0, shape[1]-1); a_m.set_ylim(shape[0]-1, 0)
                slc = slice(None, None, self.q_step)
                mask = np.sqrt(Px_all[t]**2 + Py_all[t]**2)[slc, slc] > (global_max_p * 0.05)
                q = a_m.quiver(X_grid[slc, slc][mask], Y_grid[slc, slc][mask], Px_all[t][slc, slc][mask], Py_all[t][slc, slc][mask], color='cyan', scale=fixed_scale, scale_units='width', angles='xy')
                qk = a_m.quiverkey(q, X=0.88, Y=1.06, U=ref_len, label=f'{ref_len:.1e} $mm/s$', labelpos='E', coordinates='axes', color='red', fontproperties={'size': 8})
                f_m.savefig(os.path.join(seq_out_dir, 'momentum', f'momentum_{tag}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0.05)
                q.remove(); qk.remove()

        plt.close('all')
        log_c = f"===== 序列高级物理与拓扑分析完成 =====\n输出目录: {seq_out_dir}\n处理帧数: {frames} 帧\n"
        return os.path.join(seq_out_dir, 'hfield'), self.write_log("ImageSeq", log_c, target_dir=seq_out_dir)
    
    def run_calibration(self, calib_dir, fps, in_period):
        import matplotlib
        import matplotlib.pyplot as plt
        import json
        import tkinter as tk
        from tkinter import messagebox

        fps, in_period = float(fps), float(in_period)
        if not calib_dir or not os.path.exists(calib_dir):
            raise ValueError("未选择定标总目录！请选择根目录。")

        # 自动感知数据规模
        ch_indices = []
        for d in os.listdir(calib_dir):
            if d.startswith("CH") and "_Amp" in d:
                try: ch_indices.append(int(d.split("_")[0][2:]))
                except: pass
        num_speakers = max(ch_indices) if ch_indices else 8

        levels = [0.1, 0.4, 0.7, 1.0]
        calib_out_dir = os.path.join(calib_dir, "Calibration_Results")
        os.makedirs(calib_out_dir, exist_ok=True)
        img_out_dir = os.path.join(calib_out_dir, "AutoPick_Validation_Maps")
        os.makedirs(img_out_dir, exist_ok=True)
        csv_out_dir = os.path.join(calib_out_dir, "Time_Series_Data")
        os.makedirs(csv_out_dir, exist_ok=True)

        Iref = self._read_and_crop(self.ref_path)
        
        # 阵列选择弹窗交互
        root_box = tk.Tk()
        root_box.withdraw()
        is_circle = messagebox.askyesno("阵列几何配置", "请选择您的阵列形状：\n\n【是(Yes)】：圆形阵列定标\n【否(No)】：直线阵列定标")
        root_box.destroy()

        # ----------------------------------------------------
        # 步骤 1：全图几何参考基准线标定与长条形 BBox 动态裁切
        # ----------------------------------------------------
        matplotlib.use('TkAgg', force=True)
        fig, ax = plt.subplots(figsize=(8, 8), num='全局几何基准标定')
        ax.imshow(Iref, cmap='gray')
        
        if is_circle:
            c_sel = MasterCircleSelector(ax, fig)
            plt.show(block=True)
            if c_sel.center is None: raise ValueError("操作被用户取消。")
        else:
            m_sel = MasterLineSelector(ax, fig)
            plt.show(block=True)
            if m_sel.start_pt is None: raise ValueError("操作被用户取消。")

        matplotlib.use('Agg', force=True)

        # 🌟 核心提速：基于一三线/圈，计算包含裕度的 BBox (Bounding Box)
        h_orig, w_orig = Iref.shape
        if is_circle:
            R_max = max(c_sel.radius_orig, c_sel.radius_inner) + 30
            sx1 = max(0, int(c_sel.center[0] - R_max))
            sx2 = min(w_orig, int(c_sel.center[0] + R_max))
            sy1 = max(0, int(c_sel.center[1] - R_max))
            sy2 = min(h_orig, int(c_sel.center[1] + R_max))
        else:
            xs = m_sel.orig_xdata + m_sel.sym_xdata
            ys = m_sel.orig_ydata + m_sel.sym_ydata
            sx1 = max(0, int(min(xs)) - 30)
            sx2 = min(w_orig, int(max(xs)) + 30)
            sy1 = max(0, int(min(ys)) - 30)
            sy2 = min(h_orig, int(max(ys)) + 30)

        # 全局截断所有计算域，使得 FCD 仅在这个狭长/环形的子区域内狂飙
        Iref_sub = Iref[sy1:sy2, sx1:sx2]
        mm_per_pixel = self._estimate_mm_per_pixel(Iref_sub)
        e = self.edge

        # ----------------------------------------------------
        # 步骤 2：基于目标线/圈掩膜的振幅自动寻峰定位
        # ----------------------------------------------------
        speaker_pts = []
        log_msg = f"开始自动化定标流程：提取长条/圆环局域 BBox=({sx1},{sx2}, {sy1},{sy2})\n"
        print(" 开始执行阵列靶向自动寻峰，请关注后台输出...")

        for ch in range(1, num_speakers + 1):
            folder = os.path.join(calib_dir, f"CH{ch}_Amp1.0")
            files = sorted([f for f in os.listdir(folder) if f.endswith(('.bmp', '.tiff'))])
            
            # 计算局部区域的时序高度场堆栈
            h_stack = []
            for fname in files:
                Idef = self._read_and_crop(os.path.join(folder, fname))
                Idef_sub = Idef[sy1:sy2, sx1:sx2]
                u_px, v_px = self._fcd_demodulate_correct(Iref_sub, Idef_sub)
                u_crop, v_crop = u_px[e:-e, e:-e], v_px[e:-e, e:-e]
                h_full = self._fftinvgrad(-u_crop, -v_crop) * (mm_per_pixel**2) / self.H
                h_stack.append(h_full)
                
            h_stack = np.array(h_stack)
            
            # ==========================================================
            # 🌟 新增：对定标寻峰环节引入时均场静态本底统一扣除
            # 保证提取出的 amp_map 和输出的质检监视图绝对纯净，无红蓝大斜坡
            # ==========================================================
            h_mean_time = np.mean(h_stack, axis=0)
            c_bg = self._get_bg_cache(h_mean_time.shape)
            C_coeff = c_bg['A_pinv'] @ h_mean_time.flatten()
            bg_static = (c_bg['A'] @ C_coeff).reshape(h_mean_time.shape)
            h_stack -= bg_static
            # ==========================================================
            
            # 提取该通道的全局振幅映射
            amp_map = np.std(h_stack, axis=0) * np.sqrt(2.0)
            
            # 🌟 核心：在子区域上精确复现“目标参考线/圆”，只在这条线上寻找振幅最大值
            mask = np.zeros_like(amp_map, dtype=np.uint8)
            if is_circle:
                cx_sub = int(c_sel.center[0] - sx1 - e)
                cy_sub = int(c_sel.center[1] - sy1 - e)
                r_tgt = int(c_sel.radius_target)
                cv2.circle(mask, (cx_sub, cy_sub), r_tgt, 1, thickness=4)
            else:
                tx1 = int(m_sel.target_xdata[0] - sx1 - e)
                ty1 = int(m_sel.target_ydata[0] - sy1 - e)
                tx2 = int(m_sel.target_xdata[1] - sx1 - e)
                ty2 = int(m_sel.target_ydata[1] - sy1 - e)
                cv2.line(mask, (tx1, ty1), (tx2, ty2), 1, thickness=4)
                
            amp_masked = amp_map * mask
            max_idx = np.argmax(amp_masked)
            py_sub, px_sub = np.unravel_index(max_idx, amp_map.shape)
            speaker_pts.append((px_sub, py_sub))
            log_msg += f"CH{ch} 自动寻点锁定 -> Sub_XY: ({px_sub}, {py_sub})\n"
            
            # 🌟 监视器输出：在中心帧的高度场上，叠印目标掩膜曲线和找到的靶点星号
            h_mid = h_stack[len(h_stack)//2]
            fig, ax = plt.subplots(figsize=(8,8))
            h_abs_max = np.max(np.abs(h_mid)) + 1e-6
            ax.imshow(h_mid, cmap='seismic', vmin=-h_abs_max, vmax=h_abs_max)
            ax.contour(mask, levels=[0.5], colors=['cyan'], linewidths=1.5, alpha=0.6)
            ax.plot(px_sub, py_sub, 'y*', markersize=16, markeredgecolor='black', label=f"CH{ch} 自动寻峰靶点")
            ax.set_title(f"CH{ch} 靶点自动定位质检图 (高度场叠加目标曲线)")
            ax.legend(loc='upper right')
            fig.savefig(os.path.join(img_out_dir, f"CH{ch}_AutoPick_Validation.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

        self.write_log("Auto_Pick", log_msg, target_dir=calib_out_dir)

        # ----------------------------------------------------
        # 步骤 3：纯净时域稳态振幅提取流 (仅针对靶点)
        # ----------------------------------------------------
        self.write_log("Calibration", f"正在对全网共 {num_speakers} 个阵元执行【纯净稳态振幅】拟合解调...")
        
        f_drive = 1000.0 / in_period
        omega_drive = 2.0 * np.pi * f_drive
        f_cutoff = f_drive * 0.3
        b, a = butter(4, f_cutoff, btype='high', fs=fps)

        amps_out = np.zeros((num_speakers, 4))

        for ch_idx in range(num_speakers):
            px, py = speaker_pts[ch_idx]
            
            for lvl_idx, lvl in enumerate(levels):
                folder = os.path.join(calib_dir, f"CH{ch_idx+1}_Amp{lvl:.1f}")
                files = sorted([f for f in os.listdir(folder) if f.endswith(('.bmp', '.tiff'))])
                
                N_frames = len(files)
                t_arr = np.arange(N_frames) / fps
                
                # 改为先堆叠局部高度场，用于求取当前电压层级的时均本底
                h_stack_lvl = []
                
                for i, fname in enumerate(files):
                    Idef = self._read_and_crop(os.path.join(folder, fname))
                    # 极速运算：仅仅代入狭长的 BBox 进行分析
                    Idef_sub = Idef[sy1:sy2, sx1:sx2]
                    u_px, v_px = self._fcd_demodulate_correct(Iref_sub, Idef_sub)
                    u_crop, v_crop = u_px[e:-e, e:-e], v_px[e:-e, e:-e]
                    
                    # 🌟 定标级保障：强制切断相机微震动造成的常数级积分偏移
                    u_crop -= np.mean(u_crop)
                    v_crop -= np.mean(v_crop)
                    
                    h_full = self._fftinvgrad(-u_crop, -v_crop) * (mm_per_pixel**2) / self.H
                    h_stack_lvl.append(h_full)
                
                h_stack_lvl = np.array(h_stack_lvl)
                
                # ==========================================================
                # 🌟 新增：对当前电压层级的局部时序矩阵执行静态本底统一扣除
                # 确保提取单点前，局部空间已经处于绝对零基线的完美物理状态
                # ==========================================================
                h_mean_lvl = np.mean(h_stack_lvl, axis=0)
                c_lvl = self._get_bg_cache(h_mean_lvl.shape)
                C_coeff_lvl = c_lvl['A_pinv'] @ h_mean_lvl.flatten()
                bg_static_lvl = (c_lvl['A'] @ C_coeff_lvl).reshape(h_mean_lvl.shape)
                h_stack_lvl -= bg_static_lvl
                
                # 精准剥离单像素点的时间序列 (此时已经不含任何积分漂移)
                h_arr = h_stack_lvl[:, py, px]
                
                # 执行时域滤波去噪 (进一步滤除相机散粒等高频交流微底噪)
                h_filt = filtfilt(b, a, h_arr)
                
                csv_path = os.path.join(csv_out_dir, f"CH{ch_idx+1}_Amp{lvl:.1f}_TimeSeries.csv")
                np.savetxt(csv_path, np.column_stack((t_arr, h_arr, h_filt)), 
                           delimiter=',', header="Time(s),Raw_Height(mm),Filtered_Height(mm)", comments='')

                p0 = [np.std(h_filt)*np.sqrt(2), omega_drive, 0.0, 0.0]
                bounds = ([0, omega_drive*0.95, -np.pi*2, -10], [100, omega_drive*1.05, np.pi*2, 10])
                try:
                    popt, _ = curve_fit(sine_fit_func, t_arr, h_filt, p0=p0, bounds=bounds)
                    fit_amp = abs(popt[0])
                except RuntimeError:
                    C = 2.0 * np.mean(h_filt * np.exp(-1j * omega_drive * t_arr))
                    fit_amp = np.abs(C)

                amps_out[ch_idx, lvl_idx] = fit_amp

        # ----------------------------------------------------
        # 步骤 4：生成支持多板切片的纯振幅 LUT 配置文件
        # ----------------------------------------------------
        # 强制振幅单调递增
        for ch in range(num_speakers):
            for lvl_idx in range(1, 4):
                if amps_out[ch, lvl_idx] <= amps_out[ch, lvl_idx-1]:
                    amps_out[ch, lvl_idx] = amps_out[ch, lvl_idx-1] + 1e-4

        global_max_amp = np.min(amps_out[:, 3])

        # 🌟 修复跨板映射冲突：按每块板 8 个通道切割为独立的 JSON 文件
        num_boards = (num_speakers - 1) // 8 + 1
        
        for board_idx in range(1, num_boards + 1):
            calib_data = {
                "Algorithm": "Time-Domain Sine Fitting (Amplitude Only Universal + Auto Targeting)",
                "Global_Max_Amp_mm": float(global_max_amp),
                "Target_Board": f"Board {board_idx}",
                "Speakers": {}
            }
            
            for local_ch in range(8):
                global_ch = (board_idx - 1) * 8 + local_ch
                if global_ch < num_speakers:
                    # 将全局通道映射回板内局部通道 (CH1~CH8)，完美匹配控制前端逻辑
                    calib_data["Speakers"][f"CH{local_ch+1}"] = {
                        "v_in": [0.0] + levels,
                        "amp_out": [0.0] + [float(x) for x in amps_out[global_ch]],
                        "phase_out": [0.0, 0.0, 0.0, 0.0, 0.0]
                    }

            out_json = os.path.join(calib_out_dir, f"speaker_lut_calibration_Board{board_idx}.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(calib_data, f, indent=4, ensure_ascii=False)
            
        log_c = f"===== 阵列系统【全自动寻峰定标】成功 =====\n"
        log_c += f"定标拓扑形状: {'圆形阵列' if is_circle else '直线阵列'}\n"
        log_c += f"定标换能器点数: {num_speakers} 个独立通道\n"
        log_c += f"全阵列物理无失真波高上限: {global_max_amp:.4f} mm\n"
        log_c += f"寻峰质检图已输出至: {img_out_dir}\n"
        log_c += f"阵列修正矩阵已落盘至: {out_json}\n"
        return self.write_log("Calibration", log_c, target_dir=calib_out_dir)