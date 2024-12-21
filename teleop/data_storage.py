import h5py
import time
import cv2
import numpy as np
import logging

HEAD_POSE_KEY = "head_pose"
LEFT_POSE_KEY = "left_pose"
RIGHT_POSE_KEY = "right_pose"
IMG_FRONT_KEY = "img_front"
TIME_STAMP_KEY = "time_stamp"


class Saver:

    def __init__(self):
        pass


    def create(self, filename, init_size,max_size=10000):
        self.filename = filename
        self.file = h5py.File(filename, 'w')
        self.size = init_size
        self.max_size = max_size
        self.cnt = 0

        # create dataset with resizeable shape
        self.file.create_dataset(HEAD_POSE_KEY, (init_size, 7), maxshape=(None, 7), dtype='f')
        self.file.create_dataset(LEFT_POSE_KEY, (init_size, 7), maxshape=(None, 7), dtype='f')
        self.file.create_dataset(RIGHT_POSE_KEY, (init_size, 7), maxshape=(None, 7), dtype='f')
        # timeStamp
        self.file.create_dataset(TIME_STAMP_KEY, (init_size,), maxshape=(None,), dtype='f')

        # image bytes in mem
        dt = h5py.vlen_dtype(np.dtype('uint8'))
        self.file.create_dataset(IMG_FRONT_KEY, 
                                 (init_size,), 
                                 maxshape=(None,), 
                                 compression="gzip",
                                 dtype=h5py.string_dtype())
        
        logging.info(f"Data storage created with size {init_size}.")
        

    def close(self):
        logging.info(f"Data storage saved to {self.filename}, with totally {self.cnt} samples. Time cost: {time.time()-self.file[TIME_STAMP_KEY][0]}s")
        self.file.close()
    

    def save(self, head_pose, left_pose, right_pose, image):
        '''
            store data included:
            # - head_mat (3x3)
            # - left_pose (7 elements, first 3 are position, last 4 are quaternion)
            # - right_pose (7 elements, first 3 are position, last 4 are quaternion)
        '''

        # 生成当前时间戳，精确到毫秒
        timestamp = time.time()
        logging.debug(f"current timestamp: {timestamp}")
        if(self.cnt == self.max_size):
            logging.warning("Data storage is full!")
            return
    
        if(self.cnt == self.size):
            self.size =  min(self.size*2, self.max_size)
            logging.info(f"Data storage resize to {self.size}.")
            self.file[HEAD_POSE_KEY].resize((self.size, 7))
            self.file[LEFT_POSE_KEY].resize((self.size, 7))
            self.file[RIGHT_POSE_KEY].resize((self.size, 7))
            self.file[TIME_STAMP_KEY].resize((self.size,))
            self.file[IMG_FRONT_KEY].resize((self.size,))

        self.file[TIME_STAMP_KEY][self.cnt] = timestamp
        self.file[HEAD_POSE_KEY][self.cnt] = head_pose
        self.file[LEFT_POSE_KEY][self.cnt] = left_pose
        self.file[RIGHT_POSE_KEY][self.cnt] = right_pose

        _, encoded_image = cv2.imencode('.jpg', image)
        byte_data = encoded_image.tobytes()
        print(f"types: {type(byte_data)}, size: {len(byte_data)}")
        self.file[IMG_FRONT_KEY][self.cnt] = byte_data

        self.cnt += 1

    def save_once(self, head_poses, left_poses, right_poses, images):
        size = head_poses.shape[0]
        self.file[HEAD_POSE_KEY][0:size] = head_poses
        self.file[LEFT_POSE_KEY][0:size] = left_poses
        self.file[RIGHT_POSE_KEY][0:size] = right_poses
        self.file[IMG_FRONT_KEY][0:size] = images


class Loader:

    def __init__(self, filename):
        self.filename = filename
        self.file = h5py.File(filename, 'r')
        self.times = self.file[HEAD_POSE_KEY].shape[0]
        self.index = 0

    def close(self):
        self.file.close()
    
    def load(self):
        if(self.index >= self.times):
            print("Data has been loaded!")
            return None, None, None, None
        head_pose = self.file[HEAD_POSE_KEY][self.index]
        left_pose = self.file[LEFT_POSE_KEY][self.index]
        right_pose = self.file[RIGHT_POSE_KEY][self.index]
        time_stamp = self.file[TIME_STAMP_KEY][self.index]

        data_bytes = self.file[IMG_FRONT_KEY][self.index][:]
        
        print(f"types: {type(data_bytes)}, size: {len(data_bytes)}")
        if(data_bytes is None):
            print("bytes stream over")
            return None, None, None, None
        img = cv2.imdecode(np.frombuffer(data_bytes, np.uint8), cv2.IMREAD_COLOR)

        self.index += 1
        return time_stamp, head_pose, left_pose, right_pose, img
    
    def load_once(self):
        head_pose = self.file[HEAD_POSE_KEY][:]
        left_pose = self.file[LEFT_POSE_KEY][:]
        right_pose = self.file[RIGHT_POSE_KEY][:]
        img = self.file[IMG_FRONT_KEY][:]
        return head_pose, left_pose, right_pose, img