from data_storage import Loader
import time
from se3_visualizer import visualizer
from pytransform3d import rotations

data_path= "../data_20241221-121603/"

if __name__ == '__main__':
    loader = Loader(data_path+'data_20241221-121605.h5')
    old_time= 0.0

    visualizer = visualizer()

    last_time = time.time()
    cnt = 0
    
    try:
        while True:
            
            time_stamp, head_pose, left_pose, right_pose, img = loader.load()

            head_rot = rotations.matrix_from_quaternion(head_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(head_rot,head_pose[0:3], scale=2.0)
            left_rot = rotations.matrix_from_quaternion(left_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(left_rot, left_pose[0:3], scale=5.0)
            right_rot = rotations.matrix_from_quaternion(right_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(right_rot, right_pose[0:3], scale=5.0)

            # calcualte the frequency real time
            freq = 1/(time.time() - last_time)
            last_time = time.time()
            if(cnt % 20 == 0):
                print(f"the frequency is {freq} Hz")

            cnt += 1

            if(img is not None):
                visualizer.show_img(img)
            visualizer.step()
            
    except KeyboardInterrupt:
        loader.close()
        exit(0)