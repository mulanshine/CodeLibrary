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


def threshold_predictions(predictions, thr=0.9):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


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


# def calculate_dice(pred, label):
#     n_class = 4
#     pred = pred.squeeze()
#     label = label.squeeze()
#     dice = np.array([0.0,0.0,0.0,0.0])
#     for i in range(4):
#         tpred = np.array(pred==i,dtype=np.uint8)
#         tlabel = np.array(label==i,dtype=np.uint8)
#         print(tlabel.shape)
#         print(tpred.shape)
#         if tpred.max() != 0 and tlabel.max() != 0:
#            dice[i] = mt_metrics.dice_score(tpred, tlabel)
#         else:
#            dice[i]=1
#     return dice


def calculate_dice(pred, label):
    pred = pred.squeeze()
    label = label.squeeze()
    dice = 0.0
    if label.max() == 0 and pred.max() == 0:
        dice = 1.0
    elif label.max() == 0 and pred.max() != 0:
        dice = 0.0
    elif pred.max() > 0 and label.max() > 0:
        dice = mt_metrics.dice_score(pred, label)
    elif pred.max() == 0 and label.max() > 0:
        dice = 0.
    return dice


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default="/root/chujiajia/Dataset/COVID19/")
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--imgsavepath', default="/root/chujiajia/Results/bz4_baseline_covid_epoch100/imgresult/")
parser.add_argument('--model_path', default="/root/chujiajia/Results/bz4_baseline_covid_epoch100/CP_Best.pth", type=str)
args = parser.parse_args()

# torch.cuda.set_device(int(args.gpu))
global_step = 0    

# testdataset = ACDCDataset(args.trainlist)
# testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)
testdataset = Covid19DataSet(args.dataroot, set="test",lesion_phase=False,augmentations=None)
testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)
print("testing dataset:{}".format(len(testdataset)))

model = UNet(n_channels=1, n_classes=1)
# model = torch.nn.DataParallel(model, device_ids=[0, 3])
model.cuda()
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
print("load model from" + args.model_path)
model.eval()
model.cuda()
all_dice = 0.0
c = 0
result_dice = []
mask_ids = ['XH2_10', 'XH2_70', 'XH3_26', 'XH2_89', 'XH2_34', 'XH3_7', 'XH3_12', 'XH3_123', 'XH5_36', 'XH5_49', '82', '83', '84', '85', '86', '87', '88', '89', '95', '96']
model.eval()
model.cuda()
mask_subject = defaultdict(np.ndarray)
pred_subject = defaultdict(np.ndarray)
dice = defaultdict(np.ndarray)
for mask_id in mask_ids:
    mask_subject[mask_id] = np.zeros((256,256)).reshape(1,-1)
    pred_subject[mask_id] = np.zeros((256,256)).reshape(1,-1)
    
for batch_idx, batch in enumerate(testloader):
    batch_idx = batch_idx + 1
    img, label, edge, path = batch
    image = img.numpy().copy()
    name = path[0].split("/")[-2] + "_" + path[0].split("/")[-1].split(".")[0]
    mask_id = path[0].split('/')[-2]
    if mask_id in mask_ids:
        img = torch.Tensor(img).cuda()
        model_out = F.sigmoid(model(img))
        model_out = model_out.detach().cpu().numpy() #
        model_out = threshold_predictions(model_out)
        model_out = model_out.astype(np.uint8)
        pred = model_out.squeeze()
        dsc = calculate_dice(pred,label.numpy().squeeze())
        mask_subject[mask_id] = np.hstack((mask_subject[mask_id],label.reshape(1,-1)))
        pred_subject[mask_id] = np.hstack((pred_subject[mask_id],pred.reshape(1,-1)))
        result_dice.append([path,dsc])
        if pred.max() > 0:
            print(batch_idx,path,dsc)
            pred_edge = get_edge(pred.squeeze())
            gt_edge = get_edge(label.numpy().squeeze())
            gt_edge = gt_edge.squeeze()
            gt_edge = gt_edge* label.numpy().squeeze()
            pred_edge = pred_edge * pred
            image = image.squeeze()
            image = np.array((image - image.min()) / (image.max()-image.min() + 1e-11) * 255, np.uint8)
            from PIL import Image
            image = np.array(Image.fromarray(image).convert('RGB'))
            pred_image = image.copy()
            gt_image = image.copy()
            gt_pred_img = image.copy()
            pred_image[pred_edge == 1,:] = [255,0,0]
            gt_image[gt_edge == 1,:] = [255,0,0]
            gt_pred_img[gt_edge == 1,:] = [255,0,0]
            gt_pred_img[pred_edge == 1,:] = [0,255,0]
            pred_image = Image.fromarray(np.array(pred_image,np.uint8))
            gt_image = Image.fromarray(np.array(gt_image,np.uint8))
            gt_pred_img = Image.fromarray(np.array(gt_pred_img,np.uint8))

            # name = imgname[0].split('/')[-1].split('.')[0]
            predname = name + "_pred.png" 
            labelname = name + "_gt.png" 
            gtpredname = name + "_gtpred.png" 
            imgsavepath = args.imgsavepath
            predsavepath = os.path.join(imgsavepath,predname)
            masksavepath = os.path.join(imgsavepath,labelname)
            gt_pred_spath= os.path.join(imgsavepath,gtpredname)
            gt_image.save(masksavepath)
            pred_image.save(predsavepath)
            gt_pred_img.save(gt_pred_spath)
        

c = 0
avedice =0.0# [0.0,0.0,0.0,0.0,0.0]
for mask_id in mask_ids:
    mask_subject[mask_id] = mask_subject[mask_id][0,256*256+1:]
    pred_subject[mask_id] = pred_subject[mask_id][0,256*256+1:]
    print(pred_subject[mask_id].max())
    print(mask_subject[mask_id].max())
    dice[mask_id] = calculate_dice(pred_subject[mask_id], mask_subject[mask_id])
    if dice[mask_id] != 0:
        line = "filename:{},DSC:{}\n".format(mask_id,dice[mask_id])
        print(line)
        c += 1
        avedice += dice[mask_id]

if c == 0:
    print("for C=0,set C=1")
    avedice = 0.0
    c = 1
avedice = avedice / c 

print(all_dice/c)
np.save("/root/chujiajia/Result_IMG/result_baseline.npy",result_dice)
print(result_dice.sum(dim=1))
