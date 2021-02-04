# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : weijia
import json
import os
import random
import pathlib
import pyclipper
from torch.utils import data
import glob
import torch
import numpy as np
import cv2
from dataset.augment import DataAugment
from utils.utils import draw_bbox
import torchvision.transforms as transforms
from utils.utils import ls,read_lines,remove_all,split
from PIL import Image
data_aug = DataAugment()


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是北京
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)

        if not tag:
            cv2.fillPoly(training_mask, shrinked_poly, 0)

    return score_map, training_mask


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img

def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = (int)(h * scale + 0.5)
    w = (int)(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def random_scale(img):
    h, w = img.shape[0:2]

    # base_scale = 1
    min_scale = 640.0 / min(h, w)
    # max_scale = 2000.0 / max(h, w)
    base_scale = min_scale

    random_scale = np.array([1.0, 1.0, 1.1, 1.1, 1.2, 1.4, 1.3, 1.5, 1.7, 2.0])
    scale = np.random.choice(random_scale) * base_scale

    img = scale_aligned(img, scale)
    return img

def image_label(im, text_polys, text_tags, n, m, input_size):
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = random_scale(im)

    h, w, _ = im.shape
    if text_polys.shape[0] > 0:
        text_polys = np.reshape(text_polys * ([im.shape[1], im.shape[0]] * 4),
                            (text_polys.shape[0], int(text_polys.shape[1] / 2), 2)).astype('int32')

    # 检查越界
    text_polys = check_and_validate_polys(np.array(text_polys), (h, w))
    # h, w, _ = im.shape
    # long_edge = max(h, w)
    # if long_edge > 3200:
    #     scale = 3200 / long_edge
    #     im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
    #     text_polys *= scale

    # h, w, _ = im.shape
    # short_edge = min(h, w)
    # if short_edge < input_size:
    #     scale = input_size / short_edge
    #     im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
    #     text_polys *= scale

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, i, n, m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)


    imgs = [im, training_mask]
    imgs.extend(score_maps)

    # if random.random()<0.8:
    imgs = data_aug.random_horizontal_flip(imgs)
    imgs = data_aug.random_rotate(imgs)
    imgs = data_aug.random_crop_padding(imgs, (input_size, input_size))
    # else:
    #     imgs = data_aug.resize_author(imgs, (input_size, input_size))
    im, training_mask, score_maps = imgs[0], imgs[1], imgs[2:]

    im = Image.fromarray(im)
    im = im.convert('RGB')
    im = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(im)

    im = transforms.ToTensor()(im)
    im = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(im)
    score_maps = torch.from_numpy(np.array(score_maps)).float()
    training_mask = torch.from_numpy(training_mask).float()

    return im,score_maps, training_mask

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    # lines = read_lines(gt_path)
    bboxes = []
    tags = []
    with open(gt_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = remove_all(line, '\xef\xbb\xbf')
            gt = split(line, ',')
            if gt[-1][0] == '#':
                tags.append(False)
            else:
                tags.append(True)
            box = [int(gt[i]) for i in range(8)]
            box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
            bboxes.append(box)

    return np.array(bboxes), tags

class ICDAR17(data.Dataset):
    def __init__(self, data_dir, data_shape: int = 640, n=6, m=0.5):
        self.data_shape = data_shape
        self.n = n
        self.m = m
        self.data_dir = data_dir

        ic17_mlt_train_data_dir = os.path.join(self.data_dir,'train/image')
        ic17_mlt_train_gt_dir = os.path.join(self.data_dir,'train/gt')

        data_dirs = [ic17_mlt_train_data_dir]
        gt_dirs = [ic17_mlt_train_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = ls(data_dir, '.jpg')
            img_names.extend(ls(data_dir, '.png'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = os.path.join(data_dir,img_name)
                img_paths.append(img_path)

                gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
                gt_path = os.path.join(gt_dir,gt_name)
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)

        img, score_maps, training_mask = image_label(img, bboxes, tags, n=self.n,
                                                     m=self.m,input_size=self.data_shape)

        return img, score_maps, training_mask


    def __len__(self):
        return len(self.img_paths)

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img


if __name__ == '__main__':
    import torch
    import utils.config_icdar17 as config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = ICDAR17(config.trainroot, data_shape=config.data_shape, n=config.kernel_num, m=config.min_scale)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=1)

    for i, source in enumerate(train_loader):
        img, label, mask = source
        print(label.shape)
        print(img.shape)
        print(label[0][-1].sum())
        print(mask[0].shape)
        # print(mask[0])
        show_img(((img[0].to(torch.float)).numpy().transpose(1, 2, 0)*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406]), color=True)
        show_img(label[0])
        show_img(mask[0])
        plt.show()
        break
