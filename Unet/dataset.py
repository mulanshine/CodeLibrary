import os
import os.path as osp
import numpy as np
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import math
import os 
import os.path as osp
from skimage.filters import scharr, gaussian
import json
import h5py


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pgm','.PGM','.dcm','.nrrd'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class LiverDataSet(data.Dataset):
    def __init__(self, root, set="train", augmentations=None):
        self.root = root
        self.test_id = [8, 13, 33, 39]
        self.set = set
        self.imnames = sorted(self.make_imagenames(self.root,self.set,self.test_id))
        self.valid_classes = [0, 63, 126, 189, 252]
        self.train_classes = [0, 1, 0, 0, 0]

        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        self.files = []
        self.augmentations = augmentations
    
        for im_name in self.imnames: # cropmask|cropimage
            lbl_name = im_name.replace("images","Ground").replace(".pgm",".png")
            self.files.append({
                "img": im_name,
                "lbl": lbl_name,
                "imgname":im_name,
                "maskname": lbl_name
            }) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L').resize((256,256),Image.BILINEAR)
        image = np.array(image,dtype=np.float32)
        label = Image.open(datafiles["lbl"]).convert('L').resize((256,256),Image.NEAREST)
        size = np.array(label).shape
        label = self.encode_segmap(np.array(label))

        if self.augmentations is not None:
            image, label = self.augmentations(np.asarray(image, dtype=np.uint8), np.asarray(label, dtype=np.uint8))
            
        image = image / 255.0
        image = (image - image.mean()) / (image.std() + 1e-7)

        # image = (image - image.min()) / (image.max() - image.min() + 1e-7)
        imgname = datafiles["imgname"]
        maskname = datafiles["maskname"]
        label = np.array(label,dtype=np.uint8)
        # im = gaussian(image,sigma=1)
        edge = scharr(label)
        edge[edge>0] = 1
        image = np.expand_dims(image,axis=0)
        label = np.expand_dims(np.array(label,np.float32),axis=0)
        edge = np.expand_dims(np.array(edge,np.float32),axis=0)
        return image.copy(), label.copy(), edge.copy(), imgname 

    def contour_map(self, image):
        ratio = 0.75
        image = gaussian(image,sigma=1)
        contour = scharr(image)
        contour = (contour - contour.min())/(contour.max()-contour.min()+1e-11)
        mid_v = np.sort(contour.reshape(-1))
        value = mid_v[int(ratio*len(mid_v))]
        contour[contour < value] = 0
        contour[contour >= value] = 1
        return contour
        # 75

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask


    def make_imagenames(self, root, set, test_id):
        namelist = os.listdir(root)
        imagenames = []
        print(test_id)
        for name in namelist:
            path = os.path.join(root,name,"T2SPIR/images/")
            fnames = os.listdir(path)
            if set == "train":
                if int(name) not in test_id:
                    for fname in fnames:
                        if is_image_file(fname):
                            fnamepath = osp.join(path,fname)
                            imagenames.append(fnamepath)

            elif set == "test":
                if int(name) in test_id:
                    print(name)
                    for fname in fnames:
                        print(fname)
                        if is_image_file(fname):
                            fnamepath = osp.join(path,fname)
                            imagenames.append(fnamepath)
        return imagenames


    def name(self):
        return 'LiverDataSet'


class ProstateDataSet(data.Dataset):
    # /root/chujiajia/DataSet/Prostate_Dataset/train/ProstateDx-01-0001/images/
    def __init__(self, root, set="train", augmentations=None):
        self.root = root # /root/chujiajia/DataSet/Prostate_Dataset/
        self.set = set
        self.imgdir = os.path.join(self.root,self.set) # /root/chujiajia/DataSet/Prostate_Dataset/train/
        self.imnames = sorted(self.make_imagenames(self.imgdir))
        self.files = []
        self.augmentations = augmentations
    
        for im_name in self.imnames: # cropmask|cropimage
            lbl_name = im_name.replace("images","labels")
            self.files.append({
                "img": im_name,
                "lbl": lbl_name,
                "imgname":im_name,
                "maskname": lbl_name
            }) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L').resize((256,256),Image.BILINEAR)
        image = np.array(image,dtype=np.float32)
        


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L').resize((256,256),Image.BILINEAR)
        image = np.array(image,dtype=np.float32)
        label = Image.open(datafiles["lbl"]).convert('L').resize((256,256),Image.NEAREST)
        size = np.array(label).shape
        label = self.encode_segmap(np.array(label))

        if self.augmentations is not None:
            image, label = self.augmentations(np.asarray(image, dtype=np.uint8), np.asarray(label, dtype=np.uint8))
            
        image = image / 255.0
        image = (image - image.mean()) / (image.std() + 1e-7)
        imgname = datafiles["imgname"]
        maskname = datafiles["maskname"]
        label = np.array(label,dtype=np.uint8)
        edge = scharr(label)
        edge[edge>0] = 1
        image = np.expand_dims(image,axis=0)
        label = np.expand_dims(np.array(label,np.float32),axis=0)
        edge = np.expand_dims(np.array(edge,np.float32),axis=0)
        return image.copy(), label.copy(),edge.copy(), imgname 

    def encode_segmap(self, mask):
        mask[mask>0]=1
        return mask


    # /root/chujiajia/DataSet/Prostate_Dataset/train/ProstateDx-01-0001/images/
    def make_imagenames(self, root):
        namelist = os.listdir(root)
        imagenames = []
        for name in namelist:
            path = os.path.join(root,name,"images")
            fnames = os.listdir(path)
            for fname in fnames:
                if is_image_file(fname):
                    fnamepath = osp.join(path,fname)
                    imagenames.append(fnamepath)
        return imagenames


    def name(self):
        return 'ProstateDataSet'


class ACDCDataset(data.Dataset):
    def __init__(self, data_list,augmentations=None):
        self.data_list = data_list
        self.augmentations = augmentations
        with open(data_list, 'r') as f:
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self,index):
        img = h5py.File(self.data_infos[index],'r')['image']
        gt = h5py.File(self.data_infos[index],'r')['label']
        # print(np.unique(gt))
        imgname = self.data_infos[index]

        img = np.array(img)[:,:,None].astype(np.float32).squeeze()
        gt = np.array(gt)[:,:,None].astype(np.float32).squeeze()
        image = Image.fromarray(img).convert('L')#.resize((256,256),Image.BILINEAR)
        image = np.array(image,dtype=np.float32)
        label = Image.fromarray(gt).convert('L')#.resize((256,256),Image.NEAREST)
        label = np.array(label,dtype=np.uint8)
        ori_image = image.copy()
        name = self.data_infos[index].split('/')[-1].split('.')[0] + '_img.png'
        namepath = os.path.join("/root/chujiajia/DataSet/ACDC/train/images/",name)
        # ori_image = np.array(Image.open(namepath).convert('L'))#.resize((256,256),Image.BILINEAR))

        size = np.array(label).shape
        if self.augmentations is not None:
            image, label = self.augmentations(np.asarray(image, dtype=np.uint8), np.asarray(label, dtype=np.uint8))

        image = image / image.max()
        image = (image - image.mean()) / (image.std() + 1e-7)

        edge = self.edge_map(label)
        contour = self.contour_map(image)
        image = np.expand_dims(image,axis=0)
        label = np.expand_dims(np.array(label,np.float32),axis=0)
        edge = np.expand_dims(np.array(edge,np.float32),axis=0)
        contour = np.expand_dims(np.array(contour,np.float32),axis=0)
        return image.copy(), label.copy(),edge.copy(), contour.copy(), imgname,ori_image.copy() 

    # 当前实验设定ratio为0.8不要高斯
    def contour_map(self, image):
        ratio = 0.75
        image = gaussian(image,sigma=1)
        contour = scharr(image)
        contour = (contour - contour.min())/(contour.max()-contour.min()+1e-11)
        mid_v = np.sort(contour.reshape(-1))
        value = mid_v[int(ratio*len(mid_v))]
        # value = 0.1
        # print(value)
        contour[contour < value] = 0
        contour[contour >= value] = 1
        # cont = contour * 255
        # cont = Image.fromarray(np.array(cont,np.uint8)).convert('L')
        # cont.save(os.path.join("/root/chujiajia/Results/test/",str(contour.mean())+"img.pgm"))
        return contour

    def edge_map(self, label):
        edge = scharr(label)
        edge[edge>0] = 1
        return edge


class XieHeDataset(data.Dataset):
    def __init__(self, data_list, data_root):
        self.data_list = data_list
        self.data_root = data_root

        with open(data_list, 'r') as f:
            self.data_infos = json.load(f)
    
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        path, times_idx = self.data_infos[index]
        # print(path)
        imgname = path.split('/')[-1].split(".")[0]
        # img_p = path.replace("/home/ffbian/chencheng/XieheCardiac/",self.data_root)
        img_p = os.path.join(self.data_root, path)
        # print(img_p)
        gt_p = img_p.replace("/imgs/", "/gts/")
        # /root/XieHe_DataSet/npydata/dianfen/
        # if os.path.exists(gt_p) and os.path.exists(img_p):
        img = np.load(img_p)[:, :, times_idx][:,:,None].astype(np.float32)
        gt = np.load(gt_p)[:, :, times_idx][:,:,None].astype(np.float32)
        img = img.squeeze()
        gt = gt.squeeze()
        img = np.array(img / (img.max() + 1e-11)*255,np.uint8)
        image = Image.fromarray(img).convert('L').resize((256,256),Image.BILINEAR)
        image = np.array(image,dtype=np.float32)
        label = Image.fromarray(gt).convert('L').resize((256,256),Image.NEAREST)
        label = np.array(label,dtype=np.uint8)
        # print(image.shape)
        image = image / (image.max() + 1e-11)
        image = (image - image.mean()) / (image.std() + 1e-11)

        edge = self.edge_map(label)
        contour = self.contour_map(image)
        image = np.expand_dims(image,axis=0)
        label = np.expand_dims(np.array(label, np.float32),axis=0)
        edge = np.expand_dims(np.array(edge,np.float32),axis=0)
        contour = np.expand_dims(np.array(contour,np.float32),axis=0)
        return image.copy(), label.copy(),edge.copy(), contour.copy(), imgname 


        # 当前实验设定ratio为0.8不要高斯
    def contour_map(self, image):
        ratio = 0.75
        image = gaussian(image,sigma=1)
        contour = scharr(image)
        contour = (contour - contour.min())/(contour.max()-contour.min()+1e-11)
        mid_v = np.sort(contour.reshape(-1))
        value = mid_v[int(ratio*len(mid_v))]
        # value = 0.1
        # print(value)
        contour[contour < value] = 0
        contour[contour >= value] = 1
        # cont = contour * 255
        # cont = Image.fromarray(np.array(cont,np.uint8)).convert('L')
        # cont.save(os.path.join("/root/chujiajia/Results/test/",str(contour.mean())+"img.pgm"))
        return contour

    def edge_map(self, label):
        edge = scharr(label)
        edge[edge>0] = 1
        return edge



class Covid19DataSet(data.Dataset):
    def __init__(self, root, set="train", lesion_phase=True, augmentations=None):
        self.root = root
        self.set = set
        self.phase = lesion_phase
        self.imdir = os.path.join(self.root,self.set,"labels")
        self.lbnames = sorted(self.make_imagenames(self.imdir,self.phase))
        self.files = []
        self.augmentations = augmentations
    
        for lb_name in self.lbnames: # cropmask|cropimage
            im_name = lb_name.replace("labels","images")
            self.files.append({
                "img": im_name,
                "lbl": lb_name,
                "imgname":im_name,
                "maskname": lb_name
            }) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('L').resize((256,256),Image.BILINEAR)
        image = np.array(image,dtype=np.float32)
        label = Image.open(datafiles["lbl"]).convert('L').resize((256,256),Image.NEAREST)
        size = np.array(label).shape
        label = self.encode_segmap(np.array(label))

        if self.augmentations is not None:
            image, label = self.augmentations(np.asarray(image, dtype=np.uint8), np.asarray(label, dtype=np.uint8))
            
        image = image / 255.0
        image = (image - image.mean()) / (image.std() + 1e-7)

        # image = (image - image.min()) / (image.max() - image.min() + 1e-7)
        imgname = datafiles["imgname"]
        maskname = datafiles["maskname"]
        label = np.array(label,dtype=np.uint8)
        # edge = self.contour_map(image)
        edge = 0
        image = np.expand_dims(image,axis=0)
        label = np.expand_dims(np.array(label,np.float32),axis=0)
        edge = np.expand_dims(np.array(edge,np.float32),axis=0)
        return image.copy(), label.copy(), edge.copy(), imgname 

    def contour_map(self, image):
        ratio = 0.75
        image = gaussian(image,sigma=1)
        contour = scharr(image)
        contour = (contour - contour.min())/(contour.max()-contour.min()+1e-11)
        mid_v = np.sort(contour.reshape(-1))
        value = mid_v[int(ratio*len(mid_v))]
        contour[contour < value] = 0
        contour[contour >= value] = 1
        return contour
        # 75

    def encode_segmap(self, mask):
        mask[mask>0] = 1
        return mask


    def make_imagenames(self, root, phase):
        namelist = os.listdir(root)
        imagenames = []
        for name in namelist:
            path = os.path.join(root,name)
            fnames = os.listdir(path)
            for fname in fnames:
                fnamepath = osp.join(path,fname)
                im_max = np.array(Image.open(fnamepath)).max()
                if phase: # only lesion
                    if im_max > 0 and is_image_file(fname):
                        imagenames.append(fnamepath)
                else:
                    if is_image_file(fname):
                        imagenames.append(fnamepath)
        return imagenames


    def name(self):
        return 'Covid19DataSet'


