# -*- coding: utf-8 -*-

# data config
exp_name = "Total_Text"
trainroot = '/data/glusterfs_cv_04/11121171/data/Total/'
testroot = '/data/glusterfs_cv_04/11121171/data/Total/'
workspace_dir = '/data/glusterfs_cv_04/11121171/CVPR_Text/PSENet/workspace'
workspace = ""
data_shape = 640

# train config
gpu_id = '0,1'
workers = 10
start_epoch = 0
epochs = 600

train_batch_size = 20

lr = 1e-4
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [200,400]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_input_images = False
display_output_images = False
visualization = True
display_interval = 10
show_images_interval = 50

pretrained = True
restart_training = False
checkpoint = '/data/glusterfs_cv_04/11121171/CVPR_Text/PSENet_file/Synthtext/epoch_7.0_model.pth'

# net config
backbone = 'resnet50'
Lambda = 0.7
kernel_num = 7
min_scale = 0.4
OHEM_ratio = 3
scale = 1
min_kernel_area = 10.0
# random seed
seed = 2
binary_th = 1.0


def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
