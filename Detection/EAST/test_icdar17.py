import torch
from network.model import EAST
from network.loss import Loss
import numpy as np
from PIL import Image
from tqdm import tqdm
from lib.detect import detect_17
from evaluate.script import getresult
import argparse
import os
import cv2
import shutil
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='EAST reimplementation')

# Model path
parser.add_argument('--exp_name',default= "ICDAR17", help='Where to store logs and models')
parser.add_argument('--resume', default="146_model_640aug.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--eval_path', default="/data/data_weijiawu/ICDAR17/", type=str,
                    help='the test image of target domain ')
parser.add_argument('--workspace', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/", type=str,
                    help='save model')
parser.add_argument('--vis', default=False, type=bool, help='visualization')
parser.add_argument('--is_test', default=False, type=bool, help='is test')
parser.add_argument('--vis_path', default="", type=str, help='visu')
parser.add_argument('--gt_name', default="icdar17_gt.zip", type=str, help='gt name')

args = parser.parse_args()

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def test(model,args,ther_):
    model.eval()
    if args.is_test:
        output_path = os.path.join(args.workspace, "17_submit_test")
        input_path = os.path.join(args.eval_path, "test_image")
    else:
        output_path = os.path.join(args.workspace, "17_submit")
        input_path = os.path.join(args.eval_path, "val_image")
    image_list = os.listdir(input_path)
    print("     ----------------------------------------------------------------")
    print("                           Starting Eval...")
    print("     ----------------------------------------------------------------")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # for size in range(640,2048,32):
    #     print("short_line:",size)
    for one_image in tqdm(image_list):
        image_path = os.path.join(input_path, one_image)
        img = Image.open(image_path).convert('RGB')
        orign_img = cv2.imread(image_path)
        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        if args.is_test:
            filename = filename.split("ts_")[-1]
            res_file = output_path + "/res_" + filename + '.txt'
        else:
            res_file = output_path + "/res_" + filename + '.txt'

        vis_file = args.vis_path + filename + '.jpg'
        boxes = detect_17(img, model, device)

        with open(res_file, 'w') as f:
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                points = np.reshape(poly, -1)
                if args.is_test:
                    strResult = ','.join(
                        [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                         str(points[6]), str(points[7]), str("1.0")]) + '\r\n'
                else:
                    strResult = ','.join(
                        [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                         str(points[6]), str(points[7])]) + '\r\n'

                f.write(strResult)
            if args.vis:
                for bbox in boxes:
                    # bbox = bbox / scale.repeat(int(len(bbox) / 2))
                    bbox = np.array(bbox,np.int)
                    cv2.drawContours(orign_img, [bbox[:8].reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 2)
                cv2.imwrite(vis_file, orign_img)
    if not args.is_test:
        f_score_new = getresult(output_path,args.gt_name)


if __name__ == '__main__':

    device = torch.device("cuda")
    model = EAST()
    data_parallel = False
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     data_parallel = True
    model.to(device)
    args.workspace = os.path.join(args.workspace, args.exp_name)
    args.resume = os.path.join(args.workspace,args.resume)
    print("loading pretrained model from ",args.resume)

    model.load_state_dict(torch.load(args.resume))
    for ther in range(1,10):
        ther_ = ther*0.1
        print("threshold:",ther_)
        test(model,args,ther_)







