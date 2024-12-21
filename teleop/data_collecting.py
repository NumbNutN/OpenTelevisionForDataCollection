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
import pyrealsense2 as rs

# self diy visualizer
from se3_visualizer import visualizer

from data_storage import Saver

import logging
import enum

import cv2
import os
import io
import threading

from collections import deque

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

        # here for Intel Realsense pipline
        # Configure depth and color streams
        self.cam_pipeline = rs.pipeline()
        self.cam_config = rs.config()

        self.cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # Start streaming
        self.cam_profile = self.cam_pipeline.start(self.cam_config)


        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

        self.last_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def step(self):
        
        #self.cam_playBack.resume()
        frames = self.cam_pipeline.wait_for_frames()
        # self.cam_playBack.pause()

        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process_fixed(self.tv)

        head_pose = np.concatenate([head_mat[:3, 3], rotations.quaternion_from_matrix(head_mat[:3, :3])[[1, 2, 3, 0]]])

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        # get camera frame
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        if(self.last_image == color_image).all():
            print("The image is the same!")
        
        self.last_image = color_image
        depth_image = np.asanyarray(depth_frame.get_data())

        # rotation the image 180 degree
        color_image = np.rot90(color_image, 2)

        # from bgr to rgb
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return head_pose, left_pose, right_pose, left_qpos, right_qpos, color_image
    
class Status(enum.Enum):
    INIT = 0
    COLLECTING = 1
    WAITING = 2


def visualization_thread(visualizer):
    visualizer.step()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    teleoperator = VuerTeleop('inspire_hand.yml')
    # simulator = Sim()

    visualizer = visualizer()
    saver = Saver()

    cnt = 0
    status = Status.WAITING
    last_time = time.time()

    ok = False
    first = True

    # images deque
    images = deque()

    # vis_thread = threading.Thread(target=visualization_thread, args=(visualizer,))
    # vis_thread.start()
    
    try:
        while True:
            '''
            l/r pose has a size of 7, with the first 3 elements being the position and the last 4 elements being the quaternion
            l/r qpos has a size of 12, with the first 4 elements being the quaternion of the wrist, the next 4 elements being the quaternion of the thumb, and the last 4 elements being the quaternion of the index finger
            '''
            head_pose, left_pose, right_pose, left_qpos, right_qpos, image = teleoperator.step()
            
            # left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            # np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

            head_rot = rotations.matrix_from_quaternion(head_pose[3:])[0:3, 0:3]
            
            # handle the l/r pose data
            left_rot = rotations.matrix_from_quaternion(left_pose[3:])[0:3, 0:3]
            
            # print(head_rmat)
            right_rot = rotations.matrix_from_quaternion(right_pose[3:])[0:3, 0:3]

            # print(f"head pose: {left_pose}")

            #! visualize images   ----------  decline the frequency

            visualizer.visualize_se3(head_rot, head_pose[0:3], scale=2.0)
            visualizer.visualize_se3(left_rot, left_pose[0:3], scale=5.0)
            visualizer.visualize_se3(right_rot, right_pose[0:3], scale=5.0)
            visualizer.show_img(image)
            visualizer.step()

            ok = visualizer.ok()

            # calcualte the frequency real time
            freq = 1/(time.time() - last_time)
            last_time = time.time()
            if(cnt % 20 == 0):
                print(f"the frequency is {freq} Hz")

            if(status == Status.WAITING and ok):
                if(first):
                    data_path = f'../data_{time.strftime("%Y%m%d-%H%M%S")}/'
                    os.makedirs(data_path, exist_ok=True)
                    first = False
                
                images.clear()
                # use datetime to generate a unique filename
                saver.create(data_path+ f'data_{time.strftime("%Y%m%d-%H%M%S")}.h5', 1024)
                status = Status.COLLECTING
            
            if(status == Status.COLLECTING and ok):
                # as image data is too large, we use jpg format to save the image
                #! IO operation  ----------  decline the frequency
                saver.save(head_pose, left_pose, right_pose)
                images.append(image)
                pass

            if(status == Status.COLLECTING and not ok):                
                saver.save_images_once(np.array(images))
                saver.close()
                status = Status.WAITING

            cnt += 1
            
    except KeyboardInterrupt:
        # simulator.end()
        # Saver.close()
        exit(0)
