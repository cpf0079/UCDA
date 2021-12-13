# -*- coding: utf-8 -*-

import cv2
import os
import json, random
from PIL import Image
import numpy as np
from config import cfg

num = cfg.NUM_FRAMES


def process(src, des):
    videos = os.listdir(src)
    i = 0
    for video_name in videos:
        i += 1
        print(i)
        file_name = video_name.split('.')[0]
        # print(file_name)
        vid_cap = cv2.VideoCapture(src+video_name)
        count = 0
        success, image = vid_cap.read()
        while success:
            temp = vid_cap.get(0)
            print(temp)
            count += 1
            cv2.imwrite(des + str(i) + '_' + str(count) + ".png", image)
            clc = 0.2 * 1000 * count
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, clc)
            success, image = vid_cap.read()

            if count >= num:
                break

            if temp == vid_cap.get(0):
                print("视频异常，结束循环")
                break


if __name__ == '__main__':
    src = 'D:\\cpf\\Benchmarks\\VQA\\KoNViD_1k_videos\\'
    des = 'D:\\cpf\\Benchmarks\\VQA\\frames\\'
    # des_2 = 'F:\\database\\des_2\\'
    process(src, des)