from data_storage import Loader
import time
from se3_visualizer import visualizer
from pytransform3d import rotations

data_path= "../data_20241220-174237/"

if __name__ == '__main__':
    loader = Loader(data_path+'data_20241220-174529.h5')
    old_time= 0.0

    visualizer = visualizer()

    last_time = time.time()
    cnt = 0

    head_pose, left_pose, right_pose,img = loader.load_once()

    # print sample numbers
    print(f"the number of samples is {head_pose.shape[0]}")
    
    try:
        while True:
            
            
            # print(f"the time gap is {time_stamp - old_time}")
            # print(f"the time stamp is {time_stamp}")
            # print(f"the head pose is {head_pose}")
            # print(f"the left pose is {left_pose}")
            # if(old_time and time_stamp - old_time > 0):
            #     time.sleep(time_stamp- old_time)
            # old_time = time_stamp
            # print(type(left_pose))
            if cnt >= head_pose.shape[0]:
                break
            head_rot = rotations.matrix_from_quaternion(head_pose[cnt][3:])[0:3, 0:3]
            visualizer.visualize_se3(head_rot,head_pose[cnt][0:3], scale=2.0)
            left_rot = rotations.matrix_from_quaternion(left_pose[cnt][3:])[0:3, 0:3]
            visualizer.visualize_se3(left_rot, left_pose[cnt][0:3], scale=5.0)
            right_rot = rotations.matrix_from_quaternion(right_pose[cnt][3:])[0:3, 0:3]
            visualizer.visualize_se3(right_rot, right_pose[cnt][0:3], scale=5.0)

            # calcualte the frequency real time
            freq = 1/(time.time() - last_time)
            last_time = time.time()
            if(cnt % 20 == 0):
                print(f"the frequency is {freq} Hz")

            cnt += 1

            visualizer.show_img(img[cnt])
            visualizer.step()
            
    except KeyboardInterrupt:
        loader.close()
        exit(0)