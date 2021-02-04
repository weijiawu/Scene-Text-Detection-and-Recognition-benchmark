import cv2
import os
import utils.config_totaltext as config

import shutil
import glob
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from utils.utils import write_result_as_txt,debug,load_checkpoint, save_checkpoint, setup_logger
from dataset.total_text_load import TotalTextoader
from models import PSENet
from models.loss import PSELoss
from evaluation.script import getresult
from evaluation.total_text.eval_total import evl_totaltext
from pse import decode_total as pse_decode
from dataset.total_text_load import read_mat_lindes
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def get_bboxes(gt_path):
    bboxes = []
    data = read_mat_lindes(gt_path)
    data_polygt = data['polygt']
    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])
        point_num = len(X[0])
        word = np.array(lines[4])
        if word == '#':
            continue
        arr = np.concatenate([X, Y]).T
        box = []
        for i in range(point_num):
            box.append(arr[i][0])
            box.append(arr[i][1])
        bboxes.append(box)
    return bboxes

def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def eval(model, workspace, test_path,kern_size_, device):
    model.eval()
    img_path = os.path.join(test_path, 'Images/Test/')
    gt_path = os.path.join(test_path, 'gt/Test/')

    save_path = os.path.join(workspace, 'output')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vis_ctw1500 = os.path.join(workspace, 'vis_ctw1500')
    if os.path.exists(vis_ctw1500):
        shutil.rmtree(vis_ctw1500, ignore_errors=True)
    if not os.path.exists(vis_ctw1500):
        os.makedirs(vis_ctw1500)

    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    gt_paths = [os.path.join(gt_path, 'poly_gt_' + x.split('.')[0] + '.mat') for x in os.listdir(img_path)]
    for idx,img_path in enumerate(tqdm(img_paths, desc='test models')):

        img_name = os.path.basename(img_path).split('.')[0]
        # 读取gt
        gt_path_one = gt_paths[idx]
        gt_box = get_bboxes(gt_path_one)

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        org_img = img.copy()

        img = scale_aligned_short(img)
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)

            preds_1 = torch.sigmoid(preds[0])
            preds_1 = preds_1.detach().cpu().numpy()

            preds, boxes_list = pse_decode(preds[0], config.scale,org_img,kern_size_)

        if config.visualization:
            for bbox in boxes_list:
                bbox = np.array(bbox,np.int)
                cv2.drawContours(org_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 255, 0), 2)
            for bbox in gt_box:
                bbox = np.array(bbox,np.int)
                cv2.drawContours(org_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 2)

            org_img = cv2.resize(org_img, (640, 640))

            image_list = []
            image_list.append(org_img)
            for i in range(7):
                score = (preds_1[i]*preds_1[-1]).copy().astype(np.float32)
                score = cv2.resize(score, (640, 640))
                score = np.expand_dims(score,-1)
                score = np.concatenate((score,score,score), -1)
                image_list.append(score*255)

            debug(idx, img_path, [image_list], vis_ctw1500)

        image_name = img_path.split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, boxes_list, save_path)

    #  recall precision f1
    gt_path = os.path.join(test_path, 'gt/Test')
    fid_path = os.path.join(workspace, 'res_tt.txt')
    shutil.rmtree(fid_path, ignore_errors=True)
    precision,recall,hmean = evl_totaltext(save_path,gt_path,fid_path)
    # f_score_new = getresult(save_path,config.gt_name)
    return precision,recall,hmean


if __name__ == "__main__":
    config.workspace = os.path.join(config.workspace_dir, config.exp_name)
    logger = setup_logger(os.path.join(config.workspace, 'test_log'))
    logger.info(config.print())


    # best_save_path = '{}/Best_model_0.632154.pth'.format(config.workspace)
    best_save_path = "/data/glusterfs_cv_04/11121171/CVPR_Text/PSENet_file/Total_Text/Best_model_0.787389.pth"
    # writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.kernel_num,
                   scale=config.scale)
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")
    # if num_gpus > 1:
    model = nn.DataParallel(model)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    start_epoch = load_checkpoint(best_save_path, model, logger, device, optimizer)

    for kern_size in range(3,10):
        kern_size_ = kern_size
        print("kern_size:",kern_size_)
        precision,recall,hmean = eval(model, config.workspace, config.testroot,kern_size_, device)
        print(precision,recall,hmean)