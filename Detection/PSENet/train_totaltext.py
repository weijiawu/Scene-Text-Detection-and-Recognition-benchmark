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
from torch.autograd import Variable
from utils.utils import AverageMeter
from pse import decode_total as pse_decode
from evaluation.total_text.eval_total import evl_totaltext

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks

def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,  logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    losses = AverageMeter()
    # lr = adjust_learning_rate(optimizer, epoch)
    lr = scheduler.get_lr()[0]
    for idx, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        # Forward
        outputs = net(images)

        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = labels[:, -1, :, :]
        gt_kernels = labels[:, :-1, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_mask)
        selected_masks = Variable(selected_masks.cuda())

        loss_text = criterion(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_mask.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks.cuda())
        for i in range(6):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        loss = 0.7 * loss_text + 0.3 * loss_kernel

        # loss_c, loss_s, loss = criterion(y1, labels, training_mask)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss = loss.item()
        cur_step = epoch * all_step + idx

        if idx % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, idx, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss,  batch_time, lr))
            start = time.time()

        if idx % config.show_images_interval == 0:
            if config.display_input_images:
                # show images on tensorboard
                x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
                # writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)

                show_label = labels.detach().cpu()
                b, c, h, w = show_label.size()
                show_label = show_label.reshape(b * c, h, w)
                show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=config.kernel_num, normalize=False, padding=20,
                                              pad_value=1)
                # writer.add_image(tag='input/label', img_tensor=show_label, global_step=cur_step)

            if config.display_output_images:
                y1 = torch.sigmoid(outputs)
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.kernel_num, normalize=False, padding=20, pad_value=1)
                # writer.add_image(tag='output/preds', img_tensor=show_y, global_step=cur_step)
    # writer.add_scalar(tag='Train_epoch/loss', scalar_value=train_loss / all_step, global_step=epoch)
    return train_loss / all_step, lr

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

def eval(model, workspace, test_path, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'Images/Test/')

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

    long_size = 1280
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for idx,img_path in enumerate(tqdm(img_paths, desc='test models')):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        org_img = img.copy()
        h, w = img.shape[:2]

        img = scale_aligned_short(img)

        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale,org_img)

        if config.visualization:
            for bbox in boxes_list:
                bbox = np.array(bbox,np.int)
                cv2.drawContours(org_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 255, 0), 2)

            org_img = cv2.resize(org_img, (640, 640))
            debug(idx, img_path, [[org_img]], vis_ctw1500)
        image_name = img_path.split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, boxes_list, save_path)

    # recall precision f1
    gt_path = os.path.join(test_path, 'gt/Test')
    fid_path = os.path.join(workspace, 'res_tt.txt')
    precision,recall,hmean = evl_totaltext(save_path,gt_path,fid_path)
    # f_score_new = getresult(save_path,config.gt_name)
    return precision,recall,hmean


def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def main():

    config.workspace = os.path.join(config.workspace_dir,config.exp_name)
    if config.restart_training:
        shutil.rmtree(config.workspace, ignore_errors=True)
    if not os.path.exists(config.workspace):
        os.makedirs(config.workspace)

    shutil.rmtree(os.path.join(config.workspace, 'train_log'), ignore_errors=True)

    logger = setup_logger(os.path.join(config.workspace, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")


    train_data = TotalTextoader(config.trainroot, split='train',is_transform=True,img_size=config.data_shape,
                                kernel_num=config.kernel_num,min_scale=config.min_scale)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers))

    # writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.kernel_num, scale=config.scale)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # criterion = PSELoss(Lambda=config.Lambda, ratio=config.OHEM_ratio, reduction='mean')
    criterion = dice_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
        start_epoch += 1
        # _save_path = '{}/{}'.format(config.workspace,config.checkpoint)
        # start_epoch = load_checkpoint(_save_path, model, logger, device, optimizer)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    f1 = 0
    try:
        for epoch in range(start_epoch, config.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                          logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))

            if epoch%10==0:
                save_path = '{}/epoch_{}.pth'.format(config.workspace,epoch)
                save_checkpoint(save_path, model, optimizer, epoch, logger)

            if epoch > 400:
                precision, recall, hmean = eval(model, config.workspace, config.testroot, device)
                logger.info('  ---------------------------------------')
                logger.info('     test: precision:{:.5f} recall:{:.5f}  hmean : {:.5f}'.format(precision, recall, hmean))
                logger.info('  ---------------------------------------')

                if hmean > f1:
                    f1 = hmean
                    best_save_path = '{}/Best_model_{:.6f}.pth'.format(config.workspace,f1)
                    save_checkpoint(best_save_path, model, optimizer, epoch, logger)

    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.workspace), model, optimizer, epoch, logger)



if __name__ == '__main__':
    main()
