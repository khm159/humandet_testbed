import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model import Net
from mobilenet import mobilenetv2_x1_0,mobilenetv2_x1_4
import time
from tqdm import tqdm
from osnet import osnet_x0_25,osnet_x0_5,osnet_x0_75,osnet_x1_0


def resize(im, size):
    return cv2.resize(im.astype(np.float32) / 255., size)

def preprocess(im_crops, augmentor, size):
    im_batch = torch.cat(
        [augmentor(resize(im, size)).unsqueeze(0) for im in im_crops],
        dim=0
    ).float()
    return im_batch


default_net = Net(reid=True).cuda().eval()
dummy = torch.rand([1, 3, 128, 64]).cuda()

augmentor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]),
        ])

durations = []

for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    default_net(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)

print("average time of default net : ", np.mean(
    np.asarray(durations))
    )

default_net = default_net.cpu()
del default_net
durations = []
mobilenet1 = mobilenetv2_x1_0().cuda().eval()
for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    mobilenet1(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)

print("average time of mobilenet1 : ", np.mean(
    np.asarray(durations)))

mobilenet1 = mobilenet1.cpu()
del mobilenet1

mobilenet2 = mobilenetv2_x1_4().cuda().eval()
durations = []
for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    mobilenet2(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)
print("average time of mobilenet 1.4 : ", np.mean(
    np.asarray(durations)))

osnet = osnet_x0_25(
                pretrained=True, 
                num_classes = 512
                ).cuda().eval()
durations = []
for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    osnet(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)
print("average time of osnet 0.25 : ", np.mean(
    np.asarray(durations)))

osnet = osnet_x0_5(
                pretrained=True, 
                num_classes = 512
                ).cuda().eval()
durations = []
for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    osnet(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)
print("average time of osnet 0.5 : ", np.mean(
    np.asarray(durations)))

osnet = osnet_x0_75(
                pretrained=True, 
                num_classes = 512
                ).cuda().eval()
durations = []
for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    osnet(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)
print("average time of osnet 0.75 : ", np.mean(
    np.asarray(durations)))

osnet = osnet_x1_0(
                pretrained=True, 
                num_classes = 512
                ).cuda().eval()
durations = []
for i in tqdm(range(1000)):
    # check average inference time
    start = time.time()
    osnet(dummy)
    end = time.time()
    duration = end - start
    durations.append(duration)
print("average time of osnet 1.0 : ", np.mean(
    np.asarray(durations)))