import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

class visualizer:

    def __init__(self) -> None:
        # here initialize a figure and axis
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set the aspect ratio of the plot
        self.ax.set_box_aspect([1, 1, 1])

        # add a button
        # 在图形上添加一个按钮
        self.ax_button = plt.axes([0.7, 0.01, 0.2, 0.075])  # 按钮位置
        self.button = Button(self.ax_button, "start")

        # 将按钮点击事件与回调函数绑定
        self.button.on_clicked(self.on_button_click)
        self._ok = False

    # 按钮点击事件处理函数
    def on_button_click(self, event):
        print("Start to collect data.")
        self._ok = True

    def ok(self):
        return self._ok

    # Visualize SE(3) transformation
    def visualize_so3(self, R, scale=1.0):

        self.ax.cla()
        # Plot the coordinate frame
        self.ax.quiver(0, 0, 0, R[0, 0], R[1, 0], R[2, 0], color='r', label='x')
        self.ax.quiver(0, 0, 0, R[0, 1], R[1, 1], R[2, 1], color='g', label='y')
        self.ax.quiver(0, 0, 0, R[0, 2], R[1, 2], R[2, 2], color='b', label='z')

        # Plot the origin
        self.ax.scatter(0, 0, 0, color='k')

        # Set the aspect ratio of the plot
        # self.ax.set_box_aspect([scale, scale, scale])

        # Set the axis limits
        self.ax.set_xlim([-scale, scale])
        self.ax.set_ylim([-scale, scale])
        self.ax.set_zlim([-scale, scale])

        # Set the axis labels
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        # Set the title
        # self.ax.set_title('SO(3) Transformation')

        # Set the legend
        # self.ax.legend()

    # Visualize SE(3) transformation
    def visualize_se3(self, R, t, scale=1.0):
        # Extract rotation matrix and translation vector
        # R = T[:3, :3]
        # t = T[:3, 3]

        # Plot the coordinate frame
        self.ax.quiver(t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], color='r', label='x')
        self.ax.quiver(t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], color='g', label='y')
        self.ax.quiver(t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], color='b', label='z')

        # Plot the origin
        self.ax.scatter(t[0], t[1], t[2], color='k')

        # Set the aspect ratio of the plot
        self.ax.set_box_aspect([scale, scale, scale])

        # Set the axis limits
        self.ax.set_xlim([-scale, scale])
        self.ax.set_ylim([-scale, scale])
        self.ax.set_zlim([-scale, scale])

        # Set the axis labels
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        # Set the title
        # self.ax.set_title('SE(3) Transformation')

        # Set the legend
        # self.ax.legend()

    def step(self):
        # plt.show()
        plt.pause(0.01)
        pass



