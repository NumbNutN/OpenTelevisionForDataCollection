import math
import numpy as np
import torch

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

# self diy visualizer
from se3_visualizer import visualizer

from data_storage import Saver

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process_fixed(self.tv)

        head_pose = np.concatenate([head_mat[:3, 3], rotations.quaternion_from_matrix(head_mat[:3, :3])[[1, 2, 3, 0]]])

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_pose, left_pose, right_pose, left_qpos, right_qpos

data_path= "../data/"

import cv2
if __name__ == '__main__':
    teleoperator = VuerTeleop('inspire_hand.yml')
    # simulator = Sim()

    visualizer = visualizer()

    # use time as the file name
    Saver = Saver(data_path+ f'data_{time.time()}.h5', 1000)


    try:
        while True:
            '''
            l/r pose has a size of 7, with the first 3 elements being the position and the last 4 elements being the quaternion
            l/r qpos has a size of 12, with the first 4 elements being the quaternion of the wrist, the next 4 elements being the quaternion of the thumb, and the last 4 elements being the quaternion of the index finger
            '''
            head_pose, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            # left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            # np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
            
            # print(teleoperator.img_array.shape)
            # img = teleoperator.img_array.copy()
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # print(teleoperator.img_array)

            head_rot = rotations.matrix_from_quaternion(head_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(head_rot, head_pose[0:3], scale=2.0)

            # handle the l/r pose data
            left_rot = rotations.matrix_from_quaternion(left_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(left_rot, left_pose[0:3], scale=5.0)
            # print(head_rmat)
            right_rot = rotations.matrix_from_quaternion(right_pose[3:])[0:3, 0:3]
            visualizer.visualize_se3(right_rot, right_pose[0:3], scale=5.0)
            
            if(visualizer.ok()):
                Saver.save(head_pose, left_pose, right_pose)

            visualizer.step()

            # pause for 0.01 seconds
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        # simulator.end()
        Saver.close()
        exit(0)
