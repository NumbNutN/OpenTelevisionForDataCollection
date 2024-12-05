import h5py
import time

class Saver:

    def __init__(self, filename, times):
        self.filename = filename
        self.file = h5py.File(filename, 'w')
        self.times = times
        self.remain_times = times

        self.file.create_dataset('head_mat', (times, 9), dtype='f')
        self.file.create_dataset('left_pose', (times, 7), dtype='f')
        self.file.create_dataset('right_pose', (times, 7), dtype='f')
        # timeStamp
        self.file.create_dataset('time_stamp', (times,), dtype='f')

    def close(self):
        self.file.close()
    
    # store data included:
    # - head_mat (3x3)
    # - left_pose (7 elements, first 3 are position, last 4 are quaternion)
    # - right_pose (7 elements, first 3 are position, last 4 are quaternion)
    def save(self, head_mat, left_pose, right_pose):

        # 生成当前时间戳，精确到毫秒
        timestamp = time.time()

        if(self.remain_times == 0):
            print("Data storage is full!")
            return
        
        self.file["time_stamp"][(self.times - self.remain_times)] = timestamp
        self.file["head_mat"][(self.times - self.remain_times)] = head_mat.reshape(9)
        self.file["left_pose"][(self.times - self.remain_times)] = left_pose
        self.file["right_pose"][(self.times - self.remain_times)] = right_pose
        self.remain_times -= 1

class Loader:

    def __init__(self, filename):
        self.filename = filename
        self.file = h5py.File(filename, 'r')
        self.times = self.file['head_mat'].shape[0]
        self.index = 0

    def close(self):
        self.file.close()
    
    def load(self):
        if(self.index >= self.times):
            print("Data has been loaded!")
            return None, None, None, None
        head_mat = self.file['head_mat'][self.index].reshape(3, 3)
        left_pose = self.file['left_pose'][self.index]
        right_pose = self.file['right_pose'][self.index]
        time_stamp = self.file['time_stamp'][self.index]
        self.index += 1
        return time_stamp, head_mat, left_pose, right_pose