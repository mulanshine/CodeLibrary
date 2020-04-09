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


# data_aug = Compose([
#         ElasticTransform(alpha_range=(28.0, 30.0),
#                        sigma_range=(3.5, 4.0),
#                        p=0.3),
#         RandomAffine(degrees=4.6,
#                    scale=(0.98, 1.02),
#                    translate=(0.03, 0.03)),
#         # RandomTensorChannelShift((-0.10, 0.10)),
#     ])

data_aug = None

def dice_loss(input, target):
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice = (2.0 * intersection) / (union + eps)
    return 1 - dice

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

def one_hot2dist(label):
    label = np.array(label.cpu())
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

def multiBoundaryLoss(input,label):
    C = 4
    diceLoss = 0.
    eps = 1e-7
    label = label.squeeze()
    target = np.zeros((C,label.shape[0],label.shape[1],label.shape[2])) # 5,4,256,256
    train_gt = np.array(label)
    for i in range(C):
        target[i] = np.array((train_gt == i), dtype=np.uint8)
    target = target.transpose(1,0,2,3)
    target = torch.FloatTensor(target).cuda()
    for i in range(1, C):
        diceLoss += boundary_loss(input[:,i,::].contiguous(), target[:,i,::].contiguous())
    return diceLoss/(C-1)

def multiDiceLoss(input,label):
    C = 4
    diceLoss = 0.
    eps = 1e-7
    label = label.squeeze()
    target = np.zeros((C,label.shape[0],label.shape[1],label.shape[2])) # 5,4,256,256
    train_gt = np.array(label)
    # print(label.shape)
    # print(target.shape)
    for i in range(C):
        target[i] = np.array((train_gt == i), dtype=np.uint8)
    target = target.transpose(1,0,2,3)
    target = torch.FloatTensor(target).cuda()
    for i in range(C):
        diceLoss += dice_loss(input[:,i,::].contiguous(), target[:,i,::].contiguous())
    return diceLoss/C


def multiContourLoss(input,label,contour):
    label = label * contour
    input = input * torch.Tensor(contour).cuda()
    diceLoss = multiDiceLoss(input,label)
    return diceLoss

def calculate_dice(pred, label):
    n_class = 4
    pred = pred.squeeze()
    label = label.squeeze()
    dice = np.array([0.0,0.0,0.0,0.0])
    for i in range(4):
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


# def adjust_learning_rate(optimizer, epoch, total_epoch, initial_lr,):
#     ratio = 1 - (epoch / total_epoch)
#     lr = initial_lr * ratio
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)


# def adjust_learning_rate(optimizer, epoch, total_epoch, initial_lr,):
#     # if epoch > 0 and epoch % 20 == 0:
#     lr = initial_lr * np.power(0.1,(epoch//20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def adjust_learning_rate(optimizer, epoch, total_epoch, initial_lr,):
    total_epoch = 100
    initial_lr_rampup = 50
    if epoch <= initial_lr_rampup:
        lr = initial_lr * sigmoid_rampup(epoch, initial_lr_rampup)
    else:
        lr = cosine_lr(epoch-initial_lr_rampup,
                         total_epoch-initial_lr_rampup,
                         initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
   

# def valiation_structure(model,args,logpath,testloader,imgsavepath):
#     # mean_dice = valiation_structure(model,args,"mr_test",imgsavepath,logpath)
#     model.eval()
#     model.cuda()
#     mask_subject = defaultdict(np.ndarray)
#     pred_subject = defaultdict(np.ndarray)
#     dice = defaultdict(np.ndarray)
#     mask_ids = []
    
#     for batch_idx, batch in enumerate(testloader):
#         batch_idx = batch_idx + 1
#         img, label, edge, contour, imgname = batch
#         mask_id = imgname
#         if mask_id not in mask_ids:
#             mask_ids.append(mask_id)

#         # if not mask_subject.has_key(mask_id):
#         # print(mask_id)
#         if mask_id not in mask_subject.keys():
#             mask_subject[mask_id] = np.zeros((256,256)).reshape(1,-1)
#             pred_subject[mask_id] = np.zeros((256,256)).reshape(1,-1)

#         img = torch.Tensor(img).cuda()
#         model_out = F.softmax(model(img),dim=1)
#         model_out = model_out.detach().cpu().numpy() #
#         model_out = np.argmax(model_out,axis=1)
#         model_out = model_out.astype(np.uint8)
#         pred = model_out.squeeze()
#         mask_subject[mask_id] = np.hstack((mask_subject[mask_id],label.reshape(1,-1)))
#         pred_subject[mask_id] = np.hstack((pred_subject[mask_id],pred.reshape(1,-1)))

#         # if batch_idx < 100:
#         #     predname = name.replace("image","label") + "_pred.pgm" 
#         #     labelname = name.replace("image","label") + "_orign.pgm" 
#         #     predsavepath = os.path.join(imgsavepath,predname)
#         #     masksavepath = os.path.join(imgsavepath,labelname)
#         #     predimg = np.array(encode_segmap(pred.squeeze(),filename),dtype=np.uint8)
#         #     labelimg = np.array(encode_segmap(label.squeeze(),filename),dtype=np.uint8)
#         #     predimg = Image.fromarray(predimg)
#         #     maskimg = Image.fromarray(labelimg)
#         #     predimg.save(predsavepath)
#         #     maskimg.save(masksavepath)

#     # print("################### calculate dice ...#################")     
#     c = 0
#     avedice = [0.0,0.0,0.0,0.0]
#     for mask_id in mask_ids:
#         mask_subject[mask_id] = mask_subject[mask_id][0,256*256+1:]
#         pred_subject[mask_id] = pred_subject[mask_id][0,256*256+1:]
        
#         dice[mask_id] = calculate_dice(pred_subject[mask_id], mask_subject[mask_id])
#         if sum(dice[mask_id][1:]) != 0:
#             # line = "filename：{},dice_score:{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(filename,mask_id,sum(dice[mask_id][1:])/4,dice[mask_id][0],dice[mask_id][1],dice[mask_id][2],dice[mask_id][3],dice[mask_id][4])
#             # print(line) 
#             line = "filename：{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f}".format(mask_id,sum(dice[mask_id][1:])/3.0,dice[mask_id][0],dice[mask_id][1],dice[mask_id][2],dice[mask_id][3])
#             write_log(logpath,line)
#             c += 1
#             avedice += dice[mask_id]

#     if c == 0:
#         print("for C=0,set C=1")
#         avedice = [0.0,0.0,0.0,0.0]
#         c=1
#     avedice = avedice /c 
#     # line = "filename：{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(filename,sum(avedice[1:])/4,avedice[0],avedice[1],avedice[2],avedice[3],avedice[4])
#     line = "average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f}".format(sum(avedice[1:])/3,avedice[0],avedice[1],avedice[2],avedice[3])
#     # line = "average:{}\n0:{:3f},1:{:3f}".format(avedice[1],avedice[0],avedice[1])
    
#     mean_dice = sum(avedice[1:])/3
#     print(line)
#     write_log(logpath,line)
#     np.save(os.path.join(imgsavepath,"mask.npy"),np.array(mask_subject))
#     np.save(os.path.join(imgsavepath,"pred.npy"),np.array(pred_subject))
#     return mean_dice
            

def valiation_structure(model,args,logpath,testloader):
    # mean_dice = valiation_structure(model,args,"mr_test",imgsavepath,logpath)
    model.eval()
    model.cuda()
    mask_subject = defaultdict(np.ndarray)
    pred_subject = defaultdict(np.ndarray)
    dice = defaultdict(np.ndarray)
    mask_ids = []
    
    for batch_idx, batch in enumerate(testloader):
        batch_idx = batch_idx + 1
        img, label, edge, contour, imgname = batch
        mask_id = '_'.join(imgname[0].split('/')[-1].split('_')[:3])
        if mask_id not in mask_ids:
            mask_ids.append(mask_id)

        # if not mask_subject.has_key(mask_id):
        # print(mask_id)
        if mask_id not in mask_subject.keys():
            mask_subject[mask_id] = np.zeros((256,256)).reshape(1,-1)
            pred_subject[mask_id] = np.zeros((256,256)).reshape(1,-1)

        img = torch.Tensor(img).cuda()
        model_out = F.softmax(model(img),dim=1)
        model_out = model_out.detach().cpu().numpy() #
        model_out = np.argmax(model_out,axis=1)
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
    avedice = np.array([0.0,0.0,0.0,0.0])
    for mask_id in mask_ids:
        mask_subject[mask_id] = mask_subject[mask_id][0,256*256+1:]
        pred_subject[mask_id] = pred_subject[mask_id][0,256*256+1:]
        
        dice[mask_id] = calculate_dice(pred_subject[mask_id], mask_subject[mask_id])
        if sum(dice[mask_id][1:]) != 0:
            # line = "filename：{},dice_score:{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(filename,mask_id,sum(dice[mask_id][1:])/4,dice[mask_id][0],dice[mask_id][1],dice[mask_id][2],dice[mask_id][3],dice[mask_id][4])
            # print(line) 
            line = "filename：{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f}".format(mask_id,sum(dice[mask_id][1:])/3.0,dice[mask_id][0],dice[mask_id][1],dice[mask_id][2],dice[mask_id][3])
            write_log(logpath,line)
            c += 1
            avedice += dice[mask_id]

    if c == 0:
        print("for C=0,set C=1")
        avedice = np.array([0.0,0.0,0.0,0.0])
        c=1

    avedice = avedice /c 
    # line = "filename：{},average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f},4:{:3f}".format(filename,sum(avedice[1:])/4,avedice[0],avedice[1],avedice[2],avedice[3],avedice[4])
    line = "average:{}\n0:{:3f},1:{:3f},2:{:3f},3:{:3f}".format(sum(avedice[1:])/3,avedice[0],avedice[1],avedice[2],avedice[3])
    # line = "average:{}\n0:{:3f},1:{:3f}".format(avedice[1],avedice[0],avedice[1])
    
    mean_dice = sum(avedice[1:])/3
    print(line)
    write_log(logpath,line)
    return mean_dice
            



def run_main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainlist', default="/root/chengfeng/Cardiac/sc/libs/datasets/jsonLists/train_new.json")
    # parser.add_argument('--testlist', default="/root/chengfeng/Cardiac/sc/libs/datasets/jsonLists/test_new.json")
    parser.add_argument('--testlist', default="/root/chujiajia/Code/ACDC_CF/libs/datasets/jsonLists/acdcList/Dense_TestList.json")
    parser.add_argument('--data_root', default="/root/XieHe_DataSet/")

    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--w_contour', type=float, default=1)

    parser.add_argument('--gpu', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser.add_argument('--exp', type=str, default='liver')
    parser.add_argument('--loss_term', type=str, default='dice')

    parser.add_argument('--initial_lr', type=float, default=0.001)
    parser.add_argument('--resultpath', type=str, default="/root/chujiajia/Results/")
    parser.add_argument('--experiment_name', type=str,default='dice_loss_xiehe_1')
    parser.add_argument('--num_workers', default=16, type=int)
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
    
    
    # traindataset = ACDCDataset(args.trainlist,augmentations=data_aug)
    # trainloader = DataLoader(traindataset, batch_size=args.batchsize,shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # testdataset = ACDCDataset(args.testlist)
    # testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    traindataset = XieHeDataset(args.trainlist, args.data_root)
    trainloader = DataLoader(traindataset, batch_size=args.batchsize,shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # testdataset = XieHeDataset(args.testlist, args.data_root)
    # testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)

    testdataset = ACDCDataset(args.testlist)
    testloader = DataLoader(testdataset, batch_size=1,shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("training dataset:{}".format(len(traindataset)))
    print("testing dataset:{}".format(len(testdataset)))
    # mask_ids = ["ProstateDx-03-0001","ProstateDx-03-0002","ProstateDx-03-0003","ProstateDx-03-0004","ProstateDx-03-0005"]

    model = UNet(n_channels=1, n_classes=4)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    optimizer = torch.optim.Adam(model.parameters(),lr=args.initial_lr)
    model.train()
    model.cuda()

    # Training loop
    save_mean_dice = 0.0
    for epoch in range(1, args.num_epochs + 1):
        lr = args.initial_lr
        # lr = adjust_learning_rate(optimizer, epoch-1, args.num_epochs, args.initial_lr)
        model.train()
        model.cuda()
        composite_loss_total = 0.0
        loss_dice_total = 0.0
        loss_contour_total = 0.0
        num_steps = 0
        for i, train_batch in enumerate(trainloader):
            if i < 1000:
                # Supervised component --------------------------------------------
                src_img, labels, edge, contour, train_name  = train_batch
                if epoch ==1 and i == 1:
                    print(labels.max())
                src_img = torch.Tensor(src_img).cuda()
                # print(edge)
                # edge = torch.Tensor(edge).cuda()
                # contour = torch.Tensor(contour).cuda()
                # labels = torch.Tensor(labels).cuda()
                pred_seg = F.softmax(model(src_img),dim=1)
                if args.loss_term == "edge":
                    loss_contour = multiContourLoss(pred_seg,labels,edge) * args.w_contour
                elif args.loss_term == "contour":
                    loss_contour = multiContourLoss(pred_seg,labels,contour) * args.w_contour
                elif args.loss_term == "boundary":
                    loss_contour = multiBoundaryLoss(pred_seg,labels) * args.w_contour
                else:
                    loss_contour = 0
                
                loss_dice = multiDiceLoss(pred_seg,labels)

                if epoch == 1 and i == 1:
                    print(train_name)
                    line = str(train_name)
                    write_log(logpath,line)
                
                composite_loss = loss_contour + loss_dice 

                optimizer.zero_grad()
                composite_loss.backward()
                optimizer.step()

                composite_loss_total += composite_loss.item()
                loss_dice_total += loss_dice.item()
                if args.loss_term == "edge" or args.loss_term == "contour" or args.loss_term == "boundary":
                    loss_contour_total += loss_contour.item()
                else:
                    loss_contour_total = 0

                num_steps += 1
                global_step += 1

        dice_loss_avg = loss_dice_total / num_steps
        contour_loss_avg = loss_contour_total / num_steps
        composite_loss_avg = composite_loss_total / num_steps

        torch.save(model.state_dict(),os.path.join(savepath, 'CP_latest.pth'))
        torch.save(model.state_dict(),os.path.join(savepath, 'CP'+str(epoch)+'.pth'))

        line = "Epoch: {},Composite Loss: {:.6f},Dice Loss:{:.6f},Contour Loss:{:.6f},lr:{:.6f},best_dice:{:.6f}".format(epoch,composite_loss_avg,dice_loss_avg,contour_loss_avg,lr,save_mean_dice)
        print(line)
        write_log(logpath,line)

        if  epoch > 0 and epoch % 2 == 0:
            print("valiation_structure")
            # mean_dice = valiation_structure(model,args,"mr_test",imgsavepath,logpath)
            mean_dice = valiation_structure(model,args,logpath,testloader)
            if mean_dice > save_mean_dice:
                save_mean_dice = mean_dice
                torch.save(model.state_dict(),os.path.join(savepath, 'CP_Best.pth'))
                line = 'Best Checkpoint {} saved !Mean_dice:{}'.format(epoch,save_mean_dice)
                write_log(logpath,line)


if __name__ == '__main__':
    run_main()

