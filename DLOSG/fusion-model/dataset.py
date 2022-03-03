import os
from os import listdir
from os.path import isfile, join, isdir

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

# import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision.transforms import ToPILImage

import csv


# import skvideo.io
class divingDataset(Dataset):
    def __init__(self, data_folder, data_file, label_file, diff_file,range_file, transform,
                 tcn_range=0, random=0, test=0, num_frame=16, channel=3, size=160, downsample=2, region=0, allstage=0):

        self.data_folder = data_folder
        self.transform = transform
        self.num_frame = num_frame
        self.channel = channel
        self.size = size
        self.video_name = np.load(data_file)
        self.label = np.load(label_file)
        self.diff = np.load(diff_file)
        self.random = random
        self.test = test
        self.time_range = np.load(range_file,allow_pickle=True)
        self.tcn_range = tcn_range
        self.downsample = downsample
        self.region = region
        self.allstage = allstage

    def __getitem__(self, index):
        # 要改： 同一个video加载四次 每次用不同的stage
        video_name = str(self.video_name[index][0]).zfill(3)

        v_id = self.video_name[index][0] - 1
        # 把要循环的存起来
        vid_all = v_id

        video_path = os.path.join(self.data_folder, video_name)
        video_tensor = []
        labels = []
        diffs = []
        if self.test==2:
            # print 'test in dataset'
            video_tensor, num_tensor = self.get_test_tensor(video_path, self.num_frame, self.channel, self.size)
            labels = self.label[0][self.video_name[index][0] - 1].astype(np.float32)

            return video_tensor, num_tensor, labels
        elif self.tcn_range:
            if not self.test:
                for i in range(len(self.tcn_range)):
                    tcn = self.tcn_range[i]
                    vid_range = self.time_range[v_id]
                    while len(vid_range) != 5:
                        vid_range = np.insert(vid_range, 0, 0, 0)
                    # if tcn != 5:
                    # if tcn != 5:
                    #     mid = int((vid_range[tcn - 1] + vid_range[tcn]) / 2)
                    # if len(vid_range) == 4
                    start = int(vid_range[ tcn - 1 ])
                    # else:
                    #     start = int(vid_range[5])

                    video_tensor.append(self.get_range_tensor(video_path, self.downsample, self.num_frame, self.channel, self.size,
                                                         start))

                    labels.append(self.label[0][self.video_name[index][0] - 1].astype(np.float32))
                    diffs.append(self.diff[0][self.video_name[index][0] - 1].astype(np.float32))
            if self.test ==1 :
                for i in range(len(self.tcn_range)):
                    tcn = self.tcn_range[i]
                    vid_range = self.time_range[v_id]
                    while len(vid_range) != 5:
                        vid_range = np.insert(vid_range, 0, 0, 0)
                    # if tcn != 5:
                    # if tcn != 5:
                    #     mid = int((vid_range[tcn - 1] + vid_range[tcn]) / 2)
                    # if len(vid_range) == 4
                    start = int(vid_range[tcn - 1])
                    # else:
                    #     start = int(vid_range[5])

                    video_tensor.append(
                        self.get_range_tensor_test(video_path, self.downsample, self.num_frame, self.channel, self.size,
                                              start))

                    labels.append(self.label[0][self.video_name[index][0] - 1].astype(np.float32))
                    diffs.append(self.diff[0][self.video_name[index][0] - 1].astype(np.float32))
            return video_tensor, labels, diffs
        elif self.region:
            vid_range = self.time_range[v_id]
            if self.region == 1:
                start = 0
                end = vid_range[-1] - 1
            elif self.region == 2:
                start = vid_range[1]
                end = vid_range[-1] - 1
            #	print start, end, vid_range
            video_tensor = self.get_region_tensor(video_path, start, end, self.num_frame, self.channel, self.size)
            labels = self.label[0][self.video_name[index][0] - 1].astype(np.float32)

            return video_tensor, labels
        else:
            # print 'no test in dataset'

            video_tensor = self.get_video_tensor(video_path, self.num_frame, self.channel, self.size, self.random)

            labels = self.label[0][self.video_name[index][0] - 1].astype(np.float32)

            return video_tensor, labels

    def __len__(self):
        return len(self.video_name)

    def collect_files(self, dir_name, file_ext=".jpg", sort_files=True):
        # 取所有帧
        allfiles = [os.path.join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]

        these_files = []
        for i in range(0, len(allfiles)):
            _, ext = os.path.splitext(os.path.basename(allfiles[i]))
            if ext == file_ext:
                these_files.append(allfiles[i])

        if sort_files and len(these_files) > 0:
            these_files = sorted(these_files)

        return these_files

    def get_video_tensor(self, dir, num_frame, channel, size, random):
        images = self.collect_files(dir)
        flow = torch.FloatTensor(channel, num_frame, size, size)
        if random:
            # print 'random in get_vid_tensor'
            seed = np.random.random_integers(0, len(images) - num_frame)  # random sampling
            for i in range(num_frame):
                img = Image.open(images[i + seed])
                img = img.convert('RGB')
                img = self.transform(img)
                flow[:, i, :, :] = img
        else:
            # print 'no random in get_vid_tensor'
            downsampe = []
            for i in range(0, len(images), int(len(images) / num_frame)):
                downsampe.append(i)
            downsampe = downsampe[len(downsampe) - num_frame:]

            for idx, i in enumerate(downsampe):
                img = Image.open(images[i])
                img = img.convert('RGB')
                img = self.transform(img)
                flow[:, idx, :, :] = img
        return flow

    def get_test_tensor(self, dir, num_frame, channel, size):
        images = self.collect_files(dir)
        flow = torch.FloatTensor(channel, len(images), size, size)

        for i in range(len(images)):
            img = Image.open(images[i])
            img = img.convert('RGB')
            img = self.transform(img)
            flow[:, i, :, :] = img

        num_feature = int(len(images) / num_frame)

        res = len(images) % num_frame
        downsampe = []
        for i in range(int(res / 2), len(images), int(num_frame / 2)):
            downsampe.append(i)

        all_flow = []

        for i in range(0, len(downsampe) - 2):
            vid_tensor = flow[:, downsampe[i]:downsampe[i + 2], :, :]
            all_flow.append(vid_tensor)

        return all_flow, len(downsampe) - 2

    def get_range_tensor(self, dir, downsample, num_frame, channel, size, start):
        images = self.collect_files(dir)

        seed = np.random.random_integers(-2, 2)  # random sampling 随机采样种子

        num_frame_range = downsample * num_frame
        start += seed
        if start < 0:
            start = 0
        if start + num_frame_range + 1 > len(images):
            start = int(len(images) - num_frame_range - 1)
        if start < 0:
            start = 0

        flow = torch.FloatTensor(channel, num_frame, size, size)

        # 视频中每一帧图像转换格式
        for i in range(num_frame):
            img = Image.open(images[int(i * downsample) + start])
            img = img.convert('RGB')
            img = self.transform(img)
            flow[:, i, :, :] = img

        return flow
    def get_range_tensor_test(self, dir, downsample, num_frame, channel, size, start):
        images = self.collect_files(dir)
        seed = np.random.random_integers(-2, 2)  # random sampling 随机采样种子
        num_frame_range = downsample * num_frame
        start += seed
        if start < 0:
            start = 0
        if start + num_frame_range + 1 > len(images):
            start = int(len(images) - num_frame_range - 1)
        if start < 0:
            start = 0

        flow = torch.FloatTensor(channel, num_frame, size, size)

        # 视频中每一帧图像转换格式
        for i in range(num_frame):
            img = Image.open(images[int(i * downsample) + start])
            img = img.convert('RGB')
            img = self.transform(img)
            flow[:, i, :, :] = img

        return flow
    def get_region_tensor(self, dir, start, end, num_frame, channel, size):
        images = self.collect_files(dir)
        flow = torch.FloatTensor(channel, num_frame, size, size)
        downsampe = []
        start = int(start)
        end = int(end)
        if int(end - start) <= num_frame:
            for i in range(start, int(start + num_frame - 1), 1):
                downsampe.append(i)
        else:
            for i in range(start, end, int((end - start) / num_frame)):
                downsampe.append(i)
        # 从downsampe的长度开始，往后填充16帧
        downsampe = downsampe[len(downsampe) - num_frame:]

        for idx, i in enumerate(downsampe):
            img = Image.open(images[i])
            img = img.convert('RGB')
            img = self.transform(img)
            flow[:, idx, :, :] = img
        return flow
