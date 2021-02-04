import cv2
import os
import utils.config_synthtext as config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

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

from dataset.synthtext_load import Synthtext
from models import PSENet
from models.loss import PSELoss
from utils.utils import load_checkpoint, save_checkpoint, setup_logger
from pse import decode as pse_decode
from evaluation.script import getresult


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


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,  logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    # lr = adjust_learning_rate(optimizer, epoch)
    lr = scheduler.get_lr()[0]
    for i, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        # Forward
        y1 = net(images)
        loss_c, loss_s, loss = criterion(y1, labels, training_mask)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss_c = loss_c.item()
        loss_s = loss_s.item()
        loss = loss.item()
        cur_step = epoch * all_step + i

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f}, batch_loss_c: {:.4f}, batch_loss_s: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, i, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss, loss_c, loss_s, batch_time, lr))
            start = time.time()

        if i % config.show_images_interval == 0:
            if config.display_input_images:
                # show images on tensorboard
                x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
                # writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)

                show_label = labels.detach().cpu()
                b, c, h, w = show_label.size()
                show_label = show_label.reshape(b * c, h, w)
                show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=config.n, normalize=False, padding=20,
                                              pad_value=1)
                # writer.add_image(tag='input/label', img_tensor=show_label, global_step=cur_step)

            if config.display_output_images:
                y1 = torch.sigmoid(y1)
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=False, padding=20, pad_value=1)
                # writer.add_image(tag='output/preds', img_tensor=show_y, global_step=cur_step)
    # writer.add_scalar(tag='Train_epoch/loss', scalar_value=train_loss / all_step, global_step=epoch)
    return train_loss / all_step, lr


def eval(model, save_path, test_path, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'test_image')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    long_size = 2240
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test models'):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        #if max(h, w) > long_size:
        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1
    f_score_new = getresult(save_path,config.gt_name)
    return f_score_new


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

    train_data = Synthtext(config.trainroot, data_shape=config.data_shape, n=config.n, m=config.m)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers))

    # writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.n, scale=config.scale)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(models=models, input_to_model=dummy_input)
    criterion = PSELoss(Lambda=config.Lambda, ratio=config.OHEM_ratio, reduction='mean')
    # optimizer = torch.optim.SGD(models.parameters(), lr=config.lr, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
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
            # net_save_path = '{}/PSENet_{}_loss{:.6f}.pth'.format(config.output_dir, epoch,
            #                                                                               train_loss)
            # save_checkpoint(net_save_path, models, optimizer, epoch, logger)
            if epoch < 100:
                best_save_path = '{}/epoch_{:.1f}_model.pth'.format(config.workspace, epoch)

                save_checkpoint(best_save_path, model, optimizer, epoch, logger)

                # f_score_new = eval(model, os.path.join(config.workspace, 'output'), config.testroot, device)
                # logger.info('  ---------------------------------------')
                # logger.info('     test: f_score : {:.6f}'.format(f_score_new))
                # logger.info('  ---------------------------------------')
                # net_save_path = '{}/PSENet_{}_loss{:.6f}_r{:.6f}.pth'.format(config.output_dir, epoch,
                #                                                                               train_loss,
                #                                                                               f_score_new)
                # save_checkpoint(net_save_path, model, optimizer, epoch, logger)
                # if f_score_new > f1:
                #     f1 = f_score_new
                #     best_save_path = '{}/Best_model_{:.6f}.pth'.format(config.workspace,f1)
                #
                #     save_checkpoint(best_save_path, model, optimizer, epoch, logger)


                # writer.add_scalar(tag='Test/recall', scalar_value=recall, global_step=epoch)
                # writer.add_scalar(tag='Test/precision', scalar_value=precision, global_step=epoch)
                # writer.add_scalar(tag='Test/f1', scalar_value=f1, global_step=epoch)
        # writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.workspace), model, optimizer, epoch, logger)



if __name__ == '__main__':
    main()
