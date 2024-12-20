from data_storage import Loader
import time
from se3_visualizer import visualizer
from pytransform3d import rotations

data_path= "../data/"

if __name__ == '__main__':
    loader = Loader(data_path+'data_20241220-112819.h5')
    old_time= 0

    visualizer = visualizer()
    try:
        while True:
            
            time_stamp, head_pose, left_pose, right_pose,img = loader.load()
            # print(f"the time gap is {time_stamp - old_time}")
            # print(f"the time stamp is {time_stamp}")
            print(f"the head pose is {head_pose}")
            if(old_time and time_stamp - old_time > 0):
                time.sleep(time_stamp- old_time)
            old_time = time_stamp
            # print(type(left_pose))
            if head_pose is None:
                break
            # head_rot = rotations.matrix_from_quaternion(head_pose[3:])[0:3, 0:3]
            # visualizer.visualize_se3(head_rot,head_pose[0:3], scale=2.0)
            left_rot = rotations.matrix_from_quaternion(left_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(left_rot, left_pose[0:3], scale=5.0)
            right_rot = rotations.matrix_from_quaternion(right_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(right_rot, right_pose[0:3], scale=5.0)

            visualizer.show_img(img)
            visualizer.step()
            
    except KeyboardInterrupt:
        loader.close()
        exit(0)