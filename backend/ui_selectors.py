"""FCD 波场分析系统的交互式 UI 选择器模块。

包含三个基于 matplotlib 的交互类：
- MasterCircleSelector: 圆形阵列三同心圆生成器
- MasterLineSelector: 直线阵列三平行线生成器
- InteractiveMeasurer: 交互式测距工具（点对点 & 平行线模式）
"""

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MasterCircleSelector:
    """圆形阵列专属交互类：动态三同心圆生成器 (自动定标版)"""

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
        self.ax.set_title(
            "【定标 1/2: 圆形阵列源圆周】\n依次点击 3 个点锁定最外侧喇叭所在的第一参考圆",
            color='cyan', weight='bold')

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
        if event.button == 3:  # 右键重画
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
            plt.close(self.fig)


class MasterLineSelector:
    """直线阵列专属交互类：动态三平行线生成器 (自动定标版)"""

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
        if event.button == 3:  # 右键重画
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
            plt.close(self.fig)


class InteractiveMeasurer:
    """交互式测距工具：支持点对点测距和平行线测距两种模式"""

    def __init__(self, ax, fig, cm_per_pixel):
        self.ax = ax
        self.fig = fig
        self.cm_per_pixel = cm_per_pixel

        # 记住当前图像最纯净的画幅边界坐标限制
        self.orig_xlim = self.ax.get_xlim()
        self.orig_ylim = self.ax.get_ylim()

        self.mode = 'p2p'
        self.state = 'idle'
        self.count = 0
        self.measurements = []
        self.bg = None
        self.ext_len = 8000

        self.colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', '#FF9900', '#FF0000']

        # P2P 动态对象
        self.dyn_p2p_line, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_p2p_p1, = self.ax.plot([], [], 'wo', markersize=6, visible=False)
        self.dyn_p2p_p2, = self.ax.plot([], [], 'wo', markersize=6, visible=False)
        self.dyn_p2p_txt = self.ax.text(0, 0, "", color='black', fontsize=10,
                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)

        # PL 平行线动态对象
        self.dyn_pl_l1, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_pl_l2, = self.ax.plot([], [], color='white', lw=2, ls='--', visible=False)
        self.dyn_pl_perp, = self.ax.plot([], [], color='white', lw=2, ls='-', visible=False)
        self.dyn_pl_p1, = self.ax.plot([], [], 'ws', markersize=6, visible=False)
        self.dyn_pl_p2, = self.ax.plot([], [], 'ws', markersize=6, visible=False)
        self.dyn_pl_txt = self.ax.text(0, 0, "", color='black', fontsize=10,
                                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

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
        else:
            if self.state == 'idle':
                t2 = "【左键单击】确定基准线起点"
            elif self.state == 'ref_start':
                t2 = "【移动】拉出基准线对齐波前，【左键单击】锁定"
            elif self.state == 'parallel':
                t2 = "【移动】生成平行游标，【左键单击】放置"
            elif self.state in ['adjust', 'pan_l1', 'pan_l2']:
                t2 = "【左键拖拽】任一直线可平移微调 | 【C 键】确认保存 | 【右键】撤销重画"

        self.ax.set_title(base_title + t2, fontsize=11, weight='bold', color='yellow' if self.mode == 'p2p' else 'cyan',
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3))
        self.fig.canvas.draw_idle()

    def capture_bg(self):
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        vis_states = [a.get_visible() for a in artists]
        for a in artists: a.set_visible(False)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        for a, v in zip(artists, vis_states): a.set_visible(v)

    def reset_dynamic_state(self):
        self.state = 'idle'
        artists = [self.dyn_p2p_line, self.dyn_p2p_p1, self.dyn_p2p_p2, self.dyn_p2p_txt,
                   self.dyn_pl_l1, self.dyn_pl_l2, self.dyn_pl_perp, self.dyn_pl_p1, self.dyn_pl_p2, self.dyn_pl_txt]
        for a in artists: a.set_visible(False)
        self.capture_bg()
        self.update_title()

    def commit_measurement(self):
        color = self.colors[self.count % len(self.colors)]

        if self.mode == 'p2p':
            p1, p2 = self.p2p_p1, self.p2p_p2
            dist_cm = np.hypot(p2[0]-p1[0], p2[1]-p1[1]) * self.cm_per_pixel
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
            l1_x = [p_start[0] - ux*self.ext_len, p_start[0] + ux*self.ext_len]
            l1_y = [p_start[1] - uy*self.ext_len, p_start[1] + uy*self.ext_len]
            l2_x = [p_end[0] - ux*self.ext_len, p_end[0] + ux*self.ext_len]
            l2_y = [p_end[1] - uy*self.ext_len, p_end[1] + uy*self.ext_len]

            self.ax.plot(l1_x, l1_y, color=color, lw=1.5, ls='--')
            self.ax.plot(l2_x, l2_y, color=color, lw=1.5, ls='--')
            self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color, lw=2.5)
            self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 's', color=color, markersize=5)

            cx, cy = (p_start[0]+p_end[0])/2, (p_start[1]+p_end[1])/2
            self.ax.text(cx, cy, f" #{self.count+1}: {dist_cm:.2f}cm ", color='white',
                         fontsize=10, ha='center', va='center', bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
            self.measurements.append({'id': self.count+1, 'type': 'Parallel', 'p1': p_start, 'p2': p_end, 'dist_cm': dist_cm, 'time': time.strftime("%H:%M:%S")})

        self.count += 1

        self.ax.set_xlim(self.orig_xlim)
        self.ax.set_ylim(self.orig_ylim)

        self.reset_dynamic_state()

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

    def on_key(self, event):
        if event.key in ['m', 'M']:
            self.mode = 'pl' if self.mode == 'p2p' else 'p2p'
            self.reset_dynamic_state()
        elif event.key in ['c', 'C'] and self.mode == 'pl' and self.state in ['adjust', 'pan_l1', 'pan_l2']:
            self.commit_measurement()
        elif event.key == 'enter':
            plt.close(self.fig)

    def on_press(self, event):
        if event.inaxes != self.ax: return

        if event.button == 3:
            self.reset_dynamic_state()
            return

        if event.button != 1: return

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
                self.commit_measurement()

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
