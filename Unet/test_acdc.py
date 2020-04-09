import sys
import json
import time
import os
from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils import data, model_zoo
import medicaltorch.metrics as mt_metrics
from PIL import Image
from unet import UNet
from loss import CrossEntropy2d
import torchvision as tv
import torchvision.utils as vutils
from dataset import *
import shutil
from transforms import *
from scipy.ndimage import distance_transform_edt as distance
from skimage.filters import scharr, gaussian
import argparse



def get_edge(image):
    im = np.zeros(image.shape)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            if image[i-1,j] != image[i+1,j] or image[i,j-1] != image[i,j+1]:
                im[i,j] = 1 
            elif image[i-1,j] != image[i,j-1] or image[i,j-1] != image[i-1,j]:
                im[i,j] = 1 
            elif image[i-1,j] != image[i,j+1] or image[i,j-1] != image[i+1,j]:
                im[i,j] = 1 
    return im 


def calculate_dice(pred, label):
    n_class = 4
    pred = pred.squeeze()
    label = label.squeeze()
    dice = np.array([0.0,0.0,0.0,0.0])
    for i in range(4):
        tpred = np.array(pred==i,dtype=np.uint8)
        tlabel = np.array(label==i,dtype=np.uint8)
        if tpred.max() != 0 and tlabel.max() != 0:
           dice[i] = mt_metrics.dice_score(tpred, tlabel)
        else:
           dice[i]=1
    return dice


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--testlist', default="/root/chujiajia/Code/ACDC_CF/libs/datasets/jsonLists/acdcList/Dense_TrainValList.json")
# parser.add_argument('--testlist', default="/root/chujiajia/Code/ACDC_CF/libs/datasets/jsonLists/acdcList/Dense_TestList.json")

# parser.add_argument('--testlist', default="/root/chujiajia/Code/ACDC_CF/libs/datasets/jsonLists/acdcList/Dense_TestList.json")
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--model_path', default="/root/chujiajia/Results/contour_loss_xiehe_1/CP_Best.pth", type=str)
# model_path = "/root/chujiajia/Results/new_dice_loss_acdc_3/CP_Best.pth" # baseline
# model_path = "/root/chujiajia/Results/new_contour_loss_acdc_1/CP_Best.pth" # contour
# model_path = "/root/chujiajia/Results/contour_loss_acdc_2/CP_Best.pth"
# model_path = "/root/chujiajia/Results/new_contour_loss_acdc_2/CP_Best.pth"
args = parser.parse_args()

# torch.cuda.set_device(int(args.gpu))
global_step = 0    

# testdataset = ACDCDataset(args.trainlist)
# testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)


testdataset = ACDCDataset(args.testlist)
testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)
print("testing dataset:{}".format(len(testdataset)))

model = UNet(n_channels=1, n_classes=4)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.cuda()
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
print("load model from" + args.model_path)
model.eval()
model.cuda()
all_dice = [0,0,0,0]
c = 0
result_dice = []
for batch_idx, batch in enumerate(testloader):
    batch_idx = batch_idx + 1
    img, label, gt_edge, contour, imgname, ori_image = batch
    image = img.numpy().squeeze().copy()
    img = torch.Tensor(img).cuda()
    model_out = F.softmax(model(img),dim=1)
    model_out = model_out.detach().cpu().numpy() #
    model_out = np.argmax(model_out,axis=1)
    model_out = model_out.astype(np.uint8)
    pred = model_out.squeeze()
    label[label==1] = 4
    label[label==3] = 1
    label[label==4] = 3
    dice = calculate_dice(pred, label)
    # pred_edge = scharr(pred)
    # pred_edge[pred_edge>0] = 1
    pred_edge = get_edge(pred.squeeze())
    gt_edge = get_edge(label.numpy().squeeze())
    gt_edge = gt_edge.squeeze()
    gt_edge = gt_edge* label.numpy().squeeze()
    pred_edge = pred_edge * pred
    image = ori_image.numpy().squeeze()
    print(image.max())
    # image = np.array((image - image.min()) / (image.max()-image.min() + 1e-11) * 255, np.uint8)
    from PIL import Image
    image = np.array(Image.fromarray(image).convert('RGB'))
    pred_image = image.copy()
    gt_image = image.copy()
    pred_image[pred_edge == 1,:] = [0,0,255]
    pred_image[pred_edge == 2,:] = [0,255,0]
    pred_image[pred_edge == 3,:] = [255,0,0]
    gt_image[gt_edge == 1,:] = [0,0,255]
    gt_image[gt_edge == 2,:] = [0,255,0]
    gt_image[gt_edge == 3,:] = [255,0,0]
    pred_image = Image.fromarray(np.array(pred_image,np.uint8))
    gt_image = Image.fromarray(np.array(gt_image,np.uint8))
    print(dice)
    all_dice += dice
    c = c + 1 
    name = imgname[0].split('/')[-1].split('.')[0]
    predname = name + "_pred.png" 
    labelname = name + "_gt.png" 
    imgsavepath = "/root/chujiajia/Result_IMG/contour_acdc/"  
    predsavepath = os.path.join(imgsavepath,predname)
    masksavepath = os.path.join(imgsavepath,labelname)
    gt_image.save(masksavepath)
    pred_image.save(predsavepath)
    print(masksavepath)
    result_dice.append([name,dice])

print(all_dice/c)
np.save("/root/chujiajia/Result_IMG/list/result_contour2.npy",result_dice)
    # predimg = np.array(encode_segmap(pred.squeeze(),filename),dtype=np.uint8)
    # labelimg = np.array(encode_segmap(label.squeeze(),filename),dtype=np.uint8)
    # predimg = Image.fromarray(predimg)
    # maskimg = Image.fromarray(labelimg)
    # predimg.save(predsavepath)
    # maskimg.save(masksavepath)