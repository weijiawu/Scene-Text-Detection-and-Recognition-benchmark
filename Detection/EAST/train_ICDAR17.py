import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset.icdar17_ import ICDAR17
from network.model import EAST
from network.loss import Loss
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from lib.detect import detect_17
from evaluate.script import getresult
import argparse
import os
from lib.utils import setup_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='EAST reimplementation')

# Model path
parser.add_argument('--exp_name',default= "ICDAR17", help='Where to store logs and models')
parser.add_argument('--resume', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/SynthText/synthtext_1_model.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--train_data', default='/data/data_weijiawu/ICDAR17/train_image/', type=str,
                    help='the path of training data ')
parser.add_argument('--train_gt', default="/data/data_weijiawu/ICDAR17/gt_pseudo/", type=str,
                    help='the label of training data')
parser.add_argument('--test_image', default="/data/data_weijiawu/ICDAR17/val_image/", type=str,
                    help='the data of test data')
parser.add_argument('--output_path', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/evaluate/submit/", type=str,
                    help='the predicted output of target domain')
parser.add_argument('--workspace', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/", type=str,
                    help='save model')
parser.add_argument('--gt_name', default="icdar17_gt.zip", type=str, help='gt name')
# Training strategy
parser.add_argument('--epoch_iter', default=300, type = int,
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


def train(epoch,  model, optimizer,train_loader_source,scheduler,criterion):
    model.train()
    scheduler.step()
    epoch_loss = 0
    epoch_time = time.time()

    for i, (img_target, gt_score_target, gt_geo_target, valid_map_target) in enumerate(train_loader_source):
        start_time = time.time()
        img, gt_score, gt_geo, valid_map  = img_target.to(device), gt_score_target.to(device), gt_geo_target.to(device), valid_map_target.to(device)

        pred_score, pred_geo = model(img)

        loss  = criterion(gt_score, pred_score, gt_geo, pred_geo, valid_map)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%20==0:
            logger.info('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
            epoch + 1, args.epoch_iter, i + 1, int(len(train_loader_source)), time.time() - start_time, loss.item()))

        # if i>4000 and i%1000==0:
        #     f_score = test(epoch, model, args.t_eval_path, args.t_output_path, f_score, args.save_model)
    logger.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(7200 / args.batch_size),time.time() - epoch_time))
    logger.info(time.asctime(time.localtime(time.time())))

def test(epoch,  model, input_path,output_path,f_score):
    model.eval()

    image_list = os.listdir(input_path)
    logger.info("       ----------------------------------------------------------------")
    logger.info("                    Starting Eval...")
    logger.info("       ----------------------------------------------------------------")
    for one_image in tqdm(image_list):
        image_path = os.path.join(input_path, one_image)
        img = Image.open(image_path)

        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        res_file = output_path + "res_" + filename + '.txt'

        boxes = detect_17(img, model, device)

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

    f_score_new = getresult(output_path,args.gt_name)
    try:
        if f_score_new>f_score:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(args.workspace, 'best_model_640aug.pth'.format(epoch + 1)))
            f_score = f_score_new
    except:
        print(f_score_new)

    logger.info("\n")
    logger.info("       ---------------------------------------------------------")
    logger.info("         current f_score: {:.3f} best_f_score: {:.3f}".format(f_score_new,f_score))
    logger.info("       ---------------------------------------------------------")
    return f_score



if __name__ == '__main__':
    args.workspace = os.path.join(args.workspace, args.exp_name)
    os.makedirs(args.workspace, exist_ok=True)
    logger = setup_logger(os.path.join(args.workspace, 'train_icdar17_log'))
    criterion = Loss()
    device = torch.device("cuda")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)

    # 先产生第一次的pseudo-label
    logger.info("loading pretrained model from "+args.resume)
    model.load_state_dict(torch.load(args.resume))


    # target domain
    trainset = ICDAR17(args.train_data,args.train_gt)
    train_loader_target = data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, drop_last=True)


    f_score = 0.65
    for epoch in range(args.epoch_iter):
        train( epoch, model, optimizer,train_loader_target,scheduler,criterion)
        # if epoch%10==0 and epoch>10:
        #     state_dict = model.module.state_dict() if data_parallel else model.state_dict()
        #     torch.save(state_dict, os.path.join(args.workspace, '{}_model_640aug.pth'.format(epoch)))
        # 进行target domain的eval，看看指标。
        if epoch>120:
            f_score = test(epoch, model, args.test_image, args.output_path, f_score)


