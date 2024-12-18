from data_storage import Loader
import time
from visualize_se3 import visualizer
from pytransform3d import rotations

data_path= "../data/"

if __name__ == '__main__':
    loader = Loader(data_path+'data_1733899301.1473432.h5')

    visualizer = visualizer()
    try:
        while True:
            time.sleep(0.01)
            # if(not visualizer.ok()):
            #     continue

            time_stamp, head_pose, left_pose, right_pose = loader.load()
            print(type(left_pose))
            if head_pose is None:
                break
            # head_rot = rotations.matrix_from_quaternion(head_pose[3:])[0:3, 0:3]
            # visualizer.visualize_se3(head_rot,head_pose[0:3], scale=2.0)
            left_rot = rotations.matrix_from_quaternion(left_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(left_rot, left_pose[0:3], scale=5.0)
            right_rot = rotations.matrix_from_quaternion(right_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(right_rot, right_pose[0:3], scale=5.0)
            visualizer.step()
            
    except KeyboardInterrupt:
        loader.close()
        exit(0)