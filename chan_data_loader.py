#coding=utf-8
import os
import glob
from skimage import io
import random
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
def resize_image_fixed_size(image, image_size):
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None
    new_h, new_w = image_size
    scale_h = new_h/h
    scale_w = new_w/w
    scale = min(scale_h, scale_w)
    if scale != 1:
        image = cv2.resize(image, dsize=(round(w * scale), round(h * scale)),interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    top_pad = (new_h - h) // 2
    bottom_pad = new_h - h - top_pad
    left_pad = (new_w - w) // 2
    right_pad = new_w - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image.astype(image_dtype), window, scale, padding, crop


trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(50)
        ])


class  Dataset(object):
    def __init__(self,root, image_size,is_trans=True, cls_map=None):
        self.root = root
        self.image_size = image_size
        if cls_map is None:
            self.cls_map = self.get_cls_map()
        else:
            self.cls_map = cls_map
        self.images = glob.glob(os.path.join(self.root,'*','*.*'))
        self.is_trans = is_trans
    def get_cls_map(self):
        cp = dict()
        for ix, x in enumerate(os.listdir(self.root)):
            cp[x] = ix
        return cp
    def len(self):
        return len(self.images)
    def pull_item(self,ix):
        image_pth = self.images[ix]
        lable_id = self.cls_map[image_pth.split('/')[-2]]
        ig = Image.open(image_pth)
        if self.is_trans:
            ig = trans(ig)
        ig = np.asarray(ig)
        img, _, _, _, _= resize_image_fixed_size(ig,self.image_size)
        return img, lable_id

def get_batch(batch_size, data_set,is_shuff = True,image_size=300):
    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            img,lab = data_set.pull_item(idx[index])

            if  img is None :
                index+=1
                if index >= length:
                    index = 0
                continue

            if True:
                img = (img -[123.15, 115.90, 103.06])/255

            else:
                img = ((img + [104, 117, 123])/255-0.5)*2.0

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                label = np.zeros(shape=[batch_size,],dtype=np.int32)
                images[b,:,:,:] = img
                label[b] = lab
                index=index+1
                b=b+1
            else:
                images[b, :, :, :] = img
                label[b] = lab
                index = index + 1
                b = b + 1
            if b>=batch_size:
                yield [images,label]
                b = 0
            if index>= length:
                index = 0


