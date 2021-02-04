import cv2
import os
import utils.config_icdar17 as config

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
from pse import decode_icdar17 as pse_decode
import imageio
from dataset.total_text_load import read_mat_lindes
os.environ['CUDA_VISIBLE_DEVICES'] = "1"




def readImg(im_fn):
    im = cv2.imread(im_fn)
    if im is None:
        print('{} cv2.imread failed'.format(im_fn))
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            imt = np.array(tmp)
            imt = imt[0]
            im = imt[:, :, 0:3]
    return im

def scale_image(img, short_size=2048):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    max_scale = 3200.0 / max(h, w)
    scale = min(scale, max_scale)

    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    h = (int)(h * scale + 0.5)
    w = (int)(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def eval(model, workspace, test_path,is_test, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    if is_test:
        img_path = os.path.join(test_path, 'test')
        save_path = os.path.join(workspace, 'output_test')
    else:
        img_path = os.path.join(test_path, 'val', 'image')
        save_path = os.path.join(workspace, 'output')


    gt_path = os.path.join(test_path, 'gt/Test/')

    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vis_icd17 = os.path.join(workspace, 'vis_icd17')
    if os.path.exists(vis_icd17):
        shutil.rmtree(vis_icd17, ignore_errors=True)
    if not os.path.exists(vis_icd17):
        os.makedirs(vis_icd17)

    short_size = 1600
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    gt_paths = [os.path.join(gt_path, 'poly_gt_' + x.split('.')[0] + '.mat') for x in os.listdir(img_path)]
    for idx,img_path_one in enumerate(tqdm(img_paths, desc='test models')):
        img_name = os.path.basename(img_path_one).split('.')[0]
        if is_test:
            save_name = os.path.join(save_path, 'res_' + img_name.split('ts_')[-1] + '.txt')
        else:
            save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path_one), 'file is not exists'
        img = readImg(img_path_one)
        org_img = img.copy()

        h, w = img.shape[:2]
        img = scale_image(img,short_size)

        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale,org_img,is_test)
            # scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            # if len(boxes_list):
            #     boxes_list = boxes_list / scale
        if is_test:
            np.savetxt(save_name, boxes_list.reshape(-1, 9), delimiter=',', fmt='%d')
        else:
            np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')

    # recall precision f1
    if not is_test:
        f_score_new = getresult(save_path, config.gt_name)
    return f_score_new


if __name__ == "__main__":
    config.workspace = os.path.join(config.workspace_dir, config.exp_name)
    logger = setup_logger(os.path.join(config.workspace, 'test_log'))
    logger.info(config.print())


    # best_save_path = '{}/Best_model_0.580496.pth'.format(config.workspace)
    best_save_path = '/data/glusterfs_cv_04/11121171/CVPR_Text/PSENet/workspace/ICDAR17/epoch_model.pth'

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

    # for kern_size in range(85,98,3):
    # print(kern_size)
    # kern_size_ = kern_size*0.01
    # print("kern_size:",kern_size_)
    hmean = eval(model, config.workspace, config.testroot,config.is_test, device)
