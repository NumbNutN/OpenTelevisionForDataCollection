from data_storage import Loader
import time
from visualize_se3 import visualizer
from pytransform3d import rotations

if __name__ == '__main__':
    loader = Loader('data_1733297544.9009743.h5')

    visualizer = visualizer()
    try:
        while True:
            time.sleep(0.01)
            # if(not visualizer.ok()):
            #     continue

            time_stamp, head_mat, left_pose, right_pose = loader.load()
            if head_mat is None:
                break
            
            head_mat = head_mat.reshape(3, 3)
            visualizer.visualize_so3(head_mat, scale=2.0)
            left_rot = rotations.matrix_from_quaternion(left_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(left_rot, left_pose[0:3], scale=5.0)
            right_rot = rotations.matrix_from_quaternion(right_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(right_rot, right_pose[0:3], scale=5.0)
            visualizer.step()
            
    except KeyboardInterrupt:
        loader.close()
        exit(0)