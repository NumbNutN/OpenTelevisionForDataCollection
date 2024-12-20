import numpy as np
import matplotlib.pyplot as plt
import logging

from matplotlib.widgets import Button

class visualizer:

    def __init__(self) -> None:
        # here initialize a figure and axis, the size of the figure should be 
        self.fig = plt.figure(figsize=(16,16))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set the aspect ratio of the plot
        self.ax.set_box_aspect([1, 1, 1])

        # add a button
        # 在图形上添加一个按钮
        self.ax_button = plt.axes([0.7, 0.01, 0.2, 0.075])  # 按钮位置
        self.button = Button(self.ax_button, "start")

        # Define the position and size parameters
        # on the top left corner of the figure
        # 640 x 480    
        image_xaxis = 0.0
        image_yaxis = 0.7
        image_width = 0.4
        image_height = 0.3  # Same as width since our logo is a square

        # Define the position for the image axes
        self.ax_image = self.fig.add_axes([image_xaxis,
                                image_yaxis,
                                image_width,
                                image_height]
                            )
        self.ax_image.axis('off')  # Remove axis of the image

        # 将按钮点击事件与回调函数绑定
        self.button.on_clicked(self.status_flip)
        self._ok = False

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    # 按钮点击事件处理函数
    def status_flip(self, event):
        self._ok = not self._ok

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

    def show_img(self, img):
        # Display the image
        self.ax_image.imshow(img)


    def step(self):
        # plt.show()
        plt.pause(0.01)
        self.ax.cla()
        self.ax_image.cla()  # Clear the image plot
        pass
