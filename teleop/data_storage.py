import h5py
import time
import cv2

import logging

head_pose_name = "head_pose"
left_pose_name = "left_pose"
right_pose_name = "right_pose"
img_front_name = "img_front"
time_stamp_name = "time_stamp"


class Saver:

    def __init__(self, filename, init_size,max_size=10000):
        self.filename = filename
        self.file = h5py.File(filename, 'w')
        self.size = init_size
        self.max_size = max_size
        self.cnt = 0

        # create dataset with resizeable shape
        self.file.create_dataset(head_pose_name, (init_size, 7), maxshape=(None, 7), dtype='f')
        self.file.create_dataset(left_pose_name, (init_size, 7), maxshape=(None, 7), dtype='f')
        self.file.create_dataset(right_pose_name, (init_size, 7), maxshape=(None, 7), dtype='f')
        # timeStamp
        self.file.create_dataset(time_stamp_name, (init_size,), maxshape=(None,), dtype='f')

        # image bytes in mem
        self.file.create_dataset(img_front_name, 
                                 (init_size, 480, 640, 3), 
                                 maxshape=(None, 480, 640, 3), 
                                 compression="gzip",
                                 compression_opts=8,
                                 dtype='u1')
        

    def close(self):
        logging.info(f"Data storage saved to {self.filename}, with totally {self.cnt} samples. Time cost: {time.time()-self.file[time_stamp_name][0]}s")
        self.file.close()
    
    # store data included:
    # - head_mat (3x3)
    # - left_pose (7 elements, first 3 are position, last 4 are quaternion)
    # - right_pose (7 elements, first 3 are position, last 4 are quaternion)
    def save(self, head_pose, left_pose, right_pose, image):

        # 生成当前时间戳，精确到毫秒
        timestamp = time.time()
        logging.debug(f"current timestamp: {timestamp}")
        if(self.cnt == self.max_size):
            logging.warning("Data storage is full!")
            return
    
        if(self.cnt == self.size):
            self.size =  min(self.size*2, self.max_size)
            logging.info(f"Data storage resize to {self.size}.")
            self.file[head_pose_name].resize((self.size, 7))
            self.file[left_pose_name].resize((self.size, 7))
            self.file[right_pose_name].resize((self.size, 7))
            self.file[time_stamp_name].resize((self.size,))
            self.file[img_front_name].resize((self.size, 480, 640, 3))

        self.file[time_stamp_name][self.cnt] = timestamp
        self.file[head_pose_name][self.cnt] = head_pose
        self.file[left_pose_name][self.cnt] = left_pose
        self.file[right_pose_name][self.cnt] = right_pose

        # _, encoded_image = cv2.imencode('.jpg', image)
        # byte_data = encoded_image.tobytes()
        self.file[img_front_name][self.cnt] = image

        self.cnt += 1

class Loader:

    def __init__(self, filename):
        self.filename = filename
        self.file = h5py.File(filename, 'r')
        self.times = self.file[head_pose_name].shape[0]
        self.index = 0

    def close(self):
        self.file.close()
    
    def load(self):
        if(self.index >= self.times):
            print("Data has been loaded!")
            return None, None, None, None
        head_pose = self.file[head_pose_name][self.index]
        left_pose = self.file[left_pose_name][self.index]
        right_pose = self.file[right_pose_name][self.index]
        time_stamp = self.file[time_stamp_name][self.index]

        img = self.file[img_front_name][self.index]
        self.index += 1
        return time_stamp, head_pose, left_pose, right_pose, img