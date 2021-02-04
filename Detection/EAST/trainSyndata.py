import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset.SynthText import SynthText
from network.model import EAST
from network.loss import Loss
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from lib.detect import detect
from evaluate.script import getresult
import argparse
import os
from lib.utils import setup_logger
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='EAST reimplementation')

# Model path
parser.add_argument('--exp_name',default= "SynthText", help='Where to store logs and models')
parser.add_argument('--resume', default="/data/glusterfs_cv_04/11121171/AAAI_EAST/Baseline/EAST_v1/model_save/model_epoch_826.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--t_eval_path', default="/data/data_weijiawu/Sence_Text_detection/Paper-ACCV/DomainAdaptive/ICDAR2015/EAST_v2/ICDAR15/Test/image/", type=str,
                    help='the test image of target domain ')
parser.add_argument('--t_output_path', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/evaluate/submit/", type=str,
                    help='the predicted output of target domain')
parser.add_argument('--workspace', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/", type=str,
                    help='save model')

# Training strategy
parser.add_argument('--epoch_iter', default=8000, type = int,
                    help='the max epoch iter')
parser.add_argument('--batch_size', default=8, type = int,
                    help='batch size of training')
# parser.add_argument('--cdua', default=True, type=str2bool,
#                     help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=10, type=int,
                    help='Number of workers used in dataloading')

args = parser.parse_args()


def train(epoch,  model, optimizer,train_loader_source,criterion,f_score):
    model.train()
    scheduler.step()
    epoch_loss = 0
    epoch_time = time.time()

    for i, (img_target, gt_score_target, gt_geo_target, valid_map_target) in enumerate(train_loader_source):
        start_time = time.time()

        # source domain training
        img, gt_score, gt_geo, valid_map  = img_target.to(device), gt_score_target.to(device), gt_geo_target.to(device), valid_map_target.to(device)

        pred_score, pred_geo = model(img)

        loss  = criterion(gt_score, pred_score, gt_geo, pred_geo, valid_map)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%50 == 0:
            logger.info('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, args.epoch_iter, i + 1, int(len(train_loader_source)), time.time() - start_time, loss.item()))

    logger.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(1000 / args.batch_size),time.time() - epoch_time))
    logger.info(time.asctime(time.localtime(time.time())))

def test(epoch,  model, input_path,output_path,f_score,pths_path):
    model.eval()

    image_list = os.listdir(input_path)
    logger.info("         ----------------------------------------------------------------")
    logger.info("                    Starting Eval...")
    logger.info("         ----------------------------------------------------------------")
    for one_image in tqdm(image_list):
        image_path = os.path.join(input_path, one_image)
        img = Image.open(image_path)

        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        res_file = output_path + "res_" + filename + '.txt'

        boxes = detect(img, model, device)

        with open(res_file, 'w') as f:
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                points = np.reshape(poly, -1)
                strResult = ','.join(
                    [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                     str(points[6]), str(points[7])]) + '\r\n'
                f.write(strResult)

    f_score_new = getresult(output_path)

    # if f_score_new>f_score:
    state_dict = model.module.state_dict() if data_parallel else model.state_dict()
    torch.save(state_dict, os.path.join(pths_path, 'synthtext_model.pth'.format(epoch + 1)))
    # f_score = f_score_new

    # print("\n")
    # print("         ---------------------------------------------------------")
    # print("                     best_f_score:", f_score)
    # print("         ---------------------------------------------------------")
    return f_score_new



if __name__ == '__main__':
    train_img_path = os.path.abspath('/data/data_weijiawu/SynthText')
    train_gt_path = os.path.abspath('/data/data_weijiawu/SynthText/gt.mat')

    args.workspace = os.path.join(args.workspace, args.exp_name)
    os.makedirs(args.workspace, exist_ok=True)

    logger = setup_logger(os.path.join(args.workspace, 'train_synthtext_log'))
    criterion = Loss()
    device = torch.device("cuda")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_iter // 2], gamma=0.1)

    # 先产生第一次的pseudo-label
    # logger.info("loading pretrained model from ",args.resume)
    # model.load_state_dict(torch.load(args.resume))

    trainset_ = SynthText(train_img_path, train_gt_path)
    train_loader_source = data.DataLoader(trainset_, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, drop_last=True)

    f_score = 0.6
    for epoch in range(args.epoch_iter):

        f_score = train( epoch, model, optimizer,train_loader_source,criterion,f_score)

        state_dict = model.module.state_dict() if data_parallel else model.state_dict()
        torch.save(state_dict, os.path.join(args.workspace, 'synthtext_{}_model.pth'.format(epoch + 1)))
        # 进行target domain的eval，看看指标。
        # if epoch>8:
        #     f_score = test(epoch, model, args.t_eval_path, args.t_output_path, f_score,args.save_model)



