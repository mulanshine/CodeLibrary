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
from scipy.ndimage import distance_transform_edt as distance


# def Active_Contour_Loss(y_true, y_pred): 
#     """
#     lenth term
#     """
#     x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
#     y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]
#     delta_x = x[:,:,1:,:-2]**2
#     delta_y = y[:,:,:-2,1:]**2
#     delta_u = torch.abs(delta_x + delta_y) 
#     epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
#     w = 1
#     lenth = w * torch.sum(torch.sqrt(delta_u + epsilon)) # equ.(11) in the paper
#     """
#     region term
#     """
#     C_1 = torch.ones((256, 256)).cuda()
#     C_2 = torch.zeros((256, 256)).cuda()
#     region_in = torch.abs(torch.sum(y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
#     region_out = torch.abs(torch.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper
#     lambdaP = 1 # lambda parameter could be various.
#     loss =  lenth + lambdaP * (region_in + region_out) 
#     return loss

def active_contour_loss(y_true, y_pred, weight=1): #10
  '''
  y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
  weight: scalar, length term weight.
  '''
  # length term
  delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
  delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
  
  delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
  delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
  delta_pred = torch.abs(delta_r + delta_c) 

  epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
  lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
  
  # region term
  C_in  = torch.ones_like(y_pred)
  C_out = torch.zeros_like(y_pred)

  region_in  = torch.mean(y_pred * (y_true - C_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
  region_out = torch.mean((1-y_pred) * (y_true - C_out)**2 ) 
  region = region_in + region_out
  
  loss =  weight*lenth + region

  return loss

def dice_loss(input, target):
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice = (2.0 * intersection) / (union + eps)
    return 1 - dice

# def boundary_loss(input, target):
#     # eps = 0.0001
#     from loss import SurfaceLoss
#     bound_loss = SurfaceLoss()
#     loss = bound_loss(input, target)
#     return loss

def one_hot2dist(label):
    label = np.array(label)
    C = len(label)
    dist = np.zeros_like(label)
    for c in range(C):
        posmask = label[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            dist[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return dist


def boundary_loss(probs, label):
    dist_maps = one_hot2dist(label)
    dist_maps = torch.Tensor(dist_maps).cuda()
    pc = probs.type(torch.float32)
    dc = dist_maps.type(torch.float32)
    multipled = pc * dc
    loss = multipled.mean()
    return loss

def contour_loss(input, target, edge):
    # eps = 0.0001
    eps = 1e-7
    input = input * edge
    target = target * edge
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice = (2.0 * intersection) / (union + eps)
    return 1 - dice
    
def calculate_dice(pred, label):
    n_class = 5
    pred = pred.squeeze()
    label = label.squeeze()
    dice = np.array([0.0,0.0,0.0,0.0,0.0])
    for i in range(5):
        tpred = np.array(pred==i,dtype=np.uint8)
        tlabel = np.array(label==i,dtype=np.uint8)
        dice[i] = mt_metrics.dice_score(tpred, tlabel)
    return dice


def encode_segmap(mask,filename):
    valid_classes = [0, 61, 126, 150, 246]
    train_classes = [0, 1, 2, 3, 4]
    class_map = dict(zip(train_classes, valid_classes))
    for validc in train_classes:
        mask[mask==validc] = class_map[validc]
    return mask


def threshold_predictions(predictions, thr=0.9):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def write_log(logpath,line):
    with open(logpath,"a+") as f:
        f.write(line +"\n")


def valiation_structure(model,args,logpath,testloader,mask_ids):
    # mean_dice = valiation_structure(model,args,"mr_test",imgsavepath,logpath)
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
        #/root/chujiajia/DataSet/Prostate/test/images/ProstateDx-03-0001/images/
        # /root/chujiajia/DataSet/CHAOS/1/T2SPIR/images/
        name = path[0].split("/")[-1].split(".")[0] 
        if args.exp == "liver":
            mask_id = path[0].split('/')[-4]
        else:
            mask_id = path[0].split('/')[-3]

        if mask_id in mask_ids:
            img = torch.Tensor(img).cuda()
            model_out = F.sigmoid(model(img))
            model_out = model_out.detach().cpu().numpy() #
            model_out = threshold_predictions(model_out)
            model_out = model_out.astype(np.uint8)
            pred = model_out.squeeze()
            mask_subject[mask_id] = np.hstack((mask_subject[mask_id],label.reshape(1,-1)))
            pred_subject[mask_id] = np.hstack((pred_subject[mask_id],pred.reshape(1,-1)))

            # if batch_idx < 100:
            #     predname = name.replace("image","label") + "_pred.pgm" 
            #     labelname = name.replace("image","label") + "_orign.pgm" 
            #     predsavepath = os.path.join(imgsavepath,predname)
            #     masksavepath = os.path.join(imgsavepath,labelname)
            #     predimg = np.array(encode_segmap(pred.squeeze(),filename),dtype=np.uint8)
            #     labelimg = np.array(encode_segmap(label.squeeze(),filename),dtype=np.uint8)
            #     predimg = Image.fromarray(predimg)
            #     maskimg = Image.fromarray(labelimg)
            #     predimg.save(predsavepath)
            #     maskimg.save(masksavepath)

    # print("################### calculate dice ...#################")     
    c = 0
    avedice =0.0# [0.0,0.0,0.0,0.0,0.0]
    for mask_id in mask_ids:
        mask_subject[mask_id] = mask_subject[mask_id][0,256*256+1:]
        pred_subject[mask_id] = pred_subject[mask_id][0,256*256+1:]
        
        dice[mask_id] = calculate_dice(pred_subject[mask_id], mask_subject[mask_id])
        if dice[mask_id][1] != 0:
            # line = "filename：{},dice_score:{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(filename,mask_id,sum(dice[mask_id][1:])/4,dice[mask_id][0],dice[mask_id][1],dice[mask_id][2],dice[mask_id][3],dice[mask_id][4])
            # print(line) 
            line = "filename：{},average:{}\n0:{:3f},1:{:3f}".format(mask_id,dice[mask_id][1],dice[mask_id][0],dice[mask_id][1])

            write_log(logpath,line)
            c += 1
            avedice += dice[mask_id]

    if c == 0:
        print("for C=0,set C=1")
        avedice = [0.0,0.0]
        c=1
    avedice = avedice /c 
    # line = "filename：{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(filename,sum(avedice[1:])/4,avedice[0],avedice[1],avedice[2],avedice[3],avedice[4])
    # line = "average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(sum(avedice[1:])/4,avedice[0],avedice[1],avedice[2],avedice[3],avedice[4])
    line = "average:{}\n0:{:3f},1:{:3f}".format(avedice[1],avedice[0],avedice[1])
    
    mean_dice = avedice[1]
    print(line)
    write_log(logpath,line)
    return mean_dice
            

def run_main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prostatedataroot', default="/root/chujiajia/DataSet/Prostate/")
    parser.add_argument('--liverdataroot', default="/root/chujiajia/DataSet/CHAOS/")
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--gpu', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--exp', type=str, default='prostate')
    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--resultpath', type=str, default="/root/chujiajia/Results/")
    parser.add_argument('--experiment_name', type=str,default='bz12_active_loss_prostate_epoch200')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model_path', type=str,default="None")
    parser.add_argument('--contour', type=str,default="None")
    # parser.add_argument('--model_path', type=str,default="/root/chujiajia/Results/dice_loss_liver_1/CP_Best.pth")
    # parser.add_argument('--model_path', type=str,default="/root/chujiajia/Results/dice_loss_prostate_1/CP_Best.pth")

    
    args = parser.parse_args()
    # torch.cuda.set_device(int(args.gpu))
    global_step = 0    
    savepath = os.path.join(args.resultpath,args.experiment_name)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    imgsavepath = os.path.join(savepath,"imgresult")
    if not os.path.exists(imgsavepath):
        os.mkdir(imgsavepath)
    logpath = os.path.join(savepath,"log.txt")
    
    if args.exp == "liver":
        traindataset = LiverDataSet(args.liverdataroot, set="train")
        trainloader = DataLoader(traindataset, batch_size=args.batchsize,shuffle=True, num_workers=args.num_workers, pin_memory=True)
        testdataset = LiverDataSet(args.liverdataroot, set="test")
        testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print("training dataset:{}".format(len(traindataset)))
        print("testing dataset:{}".format(len(testdataset)))
        mask_ids = ['8', '13', '33', '39']
    else:
        traindataset = ProstateDataSet(args.prostatedataroot, set="train")
        trainloader = DataLoader(traindataset, batch_size=args.batchsize,shuffle=True, num_workers=args.num_workers, pin_memory=True)
        testdataset = ProstateDataSet(args.prostatedataroot, set="test")
        testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print("training dataset:{}".format(len(traindataset)))
        print("testing dataset:{}".format(len(testdataset)))
        mask_ids = ["ProstateDx-03-0001","ProstateDx-03-0002","ProstateDx-03-0003","ProstateDx-03-0004","ProstateDx-03-0005"]


    model = UNet(n_channels=1, n_classes=1)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()
    
    if args.model_path != "None":
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print("load model from" + args.model_path)

    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.initial_lr)


    # Training loop
    save_mean_dice = 0.0
    for epoch in range(1, args.num_epochs + 1):
        lr = args.initial_lr
        model.train()
        model.cuda()
        composite_loss_total = 0.0
        loss_dice_total = 0.0
        loss_contour_total = 0.0
        num_steps = 0
        for i, train_batch in enumerate(trainloader):
            # Supervised component --------------------------------------------
            src_img, labels, edge, train_name  = train_batch
            src_img = torch.Tensor(src_img).cuda()
            # print(edge)
            edge = torch.Tensor(edge).cuda()
            label = torch.Tensor(labels).cuda()
            pred_seg = F.sigmoid(model(src_img))

            if args.contour == "active":
                loss_contour = active_contour_loss(pred_seg,label) #* args.w_contour
            elif args.contour == "boundary":
                loss_contour = boundary_loss(pred_seg,labels) 
            elif args.contour == "contour":
                loss_contour = contour_loss(pred_seg,labels,edge) 
            elif args.contour == "None":
                loss_contour = 0

            loss_dice = dice_loss(pred_seg,label)
            
            if epoch == 1 and i == 1:
                print(train_name)
                line = str(train_name)
                write_log(logpath,line)
            
             
            if args.contour == "None":
                composite_loss = loss_dice 
                composite_loss_total += 0
            else:
                composite_loss = loss_contour + loss_dice
                composite_loss_total += composite_loss.item()

            optimizer.zero_grad()
            composite_loss.backward()
            optimizer.step()

            composite_loss_total += composite_loss.item()
            loss_dice_total += loss_dice.item()
            
            if args.contour == "None":
                loss_contour_total += 0
            else:
                loss_contour_total += loss_contour.item()

            num_steps += 1
            global_step += 1

        dice_loss_avg = loss_dice_total / num_steps
        contour_loss_avg = loss_contour_total / num_steps
        composite_loss_avg = composite_loss_total / num_steps

        torch.save(model.state_dict(),os.path.join(savepath, 'CP_latest.pth'))

        # line = "Epoch: {},Composite Loss: {:.6f},Dice Loss:{:.6f},Contour Loss:{:.6f},lr:{:.6f},best_dice:{:.6f}".format(epoch,composite_loss_avg,dice_loss_avg,contour_loss_avg,lr,save_mean_dice)
        line = "Epoch: {},Composite Loss: {:.6f},Dice Loss:{:.6f},Boundary Loss:{:.6f},lr:{:.6f},best_dice:{:.6f}".format(epoch,composite_loss_avg,dice_loss_avg,contour_loss_avg,lr,save_mean_dice)
        
        print(line)
        write_log(logpath,line)

        if  epoch > 0 and epoch % 5 == 0:
            print("valiation_structure")
            # mean_dice = valiation_structure(model,args,"mr_test",imgsavepath,logpath)
            mean_dice = valiation_structure(model,args,logpath,testloader,mask_ids)
            if mean_dice > save_mean_dice:
                save_mean_dice = mean_dice
                torch.save(model.state_dict(),os.path.join(savepath, 'CP_Best.pth'))
                line = 'Best Checkpoint {} saved !Mean_dice:{}'.format(epoch,save_mean_dice)
                write_log(logpath,line)


if __name__ == '__main__':
    run_main()

