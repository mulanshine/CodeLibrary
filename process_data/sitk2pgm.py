import SimpleITK as sitk
import numpy as np
from PIL import Image
import os.path as osp

root = "/root/chengfeng/data103/all_data/data/"
impath = "/root/chujiajia/Dataset/COVID19/test/images/"
lbpath = "/root/chujiajia/Dataset/COVID19/test/labels/"
csvfile = "/root/chujiajia/Dataset/COVID19/test.csv"
filenames = open(csvfile).readlines()
namelist = []
for name in filenames:
    name = name[:-1]
    namelist.append(name)

# 1-label.nii.gz
# 1.nii.gz
def sitk_save_nii_slice_to_pgm(path,spath):
    if os.path.splitext(path)[-1] == ".gz":
        img = sitk.ReadImage(path)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr[img_arr>600] = 600 # -1200 - 600
        img_arr[img_arr<-1200] = -1200
        img = sitk.GetImageFromArray(img_arr)
        img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = np.squeeze(img_arr)
        for i in range(img_arr.shape[0]):
            savepath = os.path.join(spath,str(i).zfill(3)+".png")
            print(savepath)
            im = img_arr[i]
            im = Image.fromarray(im)
            im.save(savepath)

def sitk_label_save_nii_slice_to_pgm(path,spath):
    if os.path.splitext(path)[-1] == ".gz":
        img = sitk.ReadImage(path)
        img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = np.squeeze(img_arr)
        img_arr[img_arr>0] = 255
        for i in range(img_arr.shape[0]):
            savepath = os.path.join(spath,str(i).zfill(3)+".png")
            print(savepath)
            im = img_arr[i]
            im = Image.fromarray(im)
            im.save(savepath)


for name in namelist:
    imname = name + ".nii.gz"
    lbname = name + "-label.nii.gz"
    path = os.path.join(root,imname)
    spath = os.path.join(impath,name)
    if not os.path.exists(spath):
        os.mkdir(spath)
    sitk_save_nii_slice_to_pgm(path,spath)
    
    lb_path = os.path.join(root,lbname)
    lbspath = os.path.join(lbpath,name)
    if not os.path.exists(lbspath):
        os.mkdir(lbspath)
    sitk_label_save_nii_slice_to_pgm(lb_path,lbspath)
    
#########################Liver segmentation############################
# def sitk_save_nii_slice_to_pgm(path,spath):
#     imgLists = os.listdir(path)
#     for name in imgLists:
#         if os.path.splitext(name)[-1] == ".dcm":
#             img = sitk.ReadImage(os.path.join(path,name))
#             img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
#             sname = name.split(".")[0]
#             img = sitk.GetArrayFromImage(img)
#             img = np.squeeze(img)
#             savepath =os.path.join(spath,sname+".pgm")
#             print(savepath)
#             img = Image.fromarray(img)
#             img.save(savepath)

# rootdir = "/root/chujiajia/DataSet/CHAOS/"
# namelist = os.listdir(rootdir)
# for name in namelist:
#     path = rootdir + name +"/T2SPIR/DICOM_anon/"
#     spath = os.path.join("/".join(path.split('/')[:-2]),"images")
#     if not os.path.exists(spath):
#         os.mkdir(spath)
#     sitk_save_nii_slice_to_pgm(path,spath)


#########################Prostate segmentation image############################
# def sitk_save_nii_slice_to_pgm(path,spath):
#     imgLists = os.listdir(path)
#     for name in imgLists:
#         if os.path.splitext(path)[-1] == ".nrrd":
#         img = sitk.ReadImage(os.path.join(path,name))
#         img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
#         sname = name.split(".")[0]
#         img = sitk.GetArrayFromImage(img)
#         img = np.squeeze(img)
#         savepath =os.path.join(spath,sname+".pgm")
#         print(savepath)
#         img = Image.fromarray(img)
#         img.save(savepath)

# rootdir = "/root/chujiajia/DataSet/Prostate/Test/images/"
# namelist = os.listdir(rootdir)
# for name in namelist:
#     path = os.path.join(rootdir, name)
#     spath = os.path.join(path,"images")
#     if not os.path.exists(spath):
#         os.mkdir(spath)
#     sitk_save_nii_slice_to_pgm(path,spath)

#########################Prostate segmentation label############################
# def sitk_save_nii_slice_to_pgm(path,spath):
#     if os.path.splitext(path)[-1] == ".nrrd":
#         img = sitk.ReadImage(path)
#         # print(0000000000000000000)
#         img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
#         img_arr = sitk.GetArrayFromImage(img)
#         img_arr = np.squeeze(img_arr)
#         # print(1111111111111111111)
#         for i in range(img_arr.shape[0]):
#             savepath =os.path.join(spath,str(i).zfill(6)+".pgm")
#             print(savepath)
#             im = img_arr[i]
#             # img = Image.fromarray(img).rotate(-180)
#             im = Image.fromarray(im)
#             im.save(savepath)

# rootdir = "/root/chujiajia/DataSet/Prostate/Test/labels/"
# namelist = os.listdir(rootdir)
# for name in namelist:
#     path = os.path.join(rootdir, name)
#     dirname = name.split('.')[0]
#     spath = os.path.join(rootdir,dirname)
#     if not os.path.exists(spath):
#         os.mkdir(spath)
#         print(spath)
#     path = "/root/chujiajia/DataSet/ZIPData/ProstateDx-03-0002_truth.nrrd"
#     spath = "/root/chujiajia/DataSet/Prostate_Dataset/test/ProstateDx-03-0002/labels/"
#     sitk_save_nii_slice_to_pgm(path,spath)

######################## cine #########################################
# def sitk_save_nii_slice_to_pgm(path,spath):
#     imgLists = os.listdir(path)
#     for name in imgLists:
#         if 'VERSION' not in name:
#             img = sitk.ReadImage(os.path.join(path,name))
#             img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
#             sname = name.split(".")[0]
#             img = sitk.GetArrayFromImage(img)
#             img = np.squeeze(img)
#             savepath =os.path.join(spath,sname+".png")
#             print(savepath)
#             img = Image.fromarray(img)
#             img.save(savepath)

# pathlist = ["/root/chujiajia/DataSet/test/cine/1/A1KZALKJ/BXTQ5U4U/",
# "/root/chujiajia/DataSet/test/cine/2/A1KZALKJ/BXTQ4E4U/",
# "/root/chujiajia/DataSet/test/cine/3/A1KZALKJ/BXTQ4U4U/",
# "/root/chujiajia/DataSet/test/cine/4/A1KZALKJ/BXTQ3E4U/",
# "/root/chujiajia/DataSet/test/cine/5/A1KZALKJ/BXTQ3U4U/",
# "/root/chujiajia/DataSet/test/cine/6/A1KZALKJ/BXTQ2E4U/",
# "/root/chujiajia/DataSet/test/cine/7/A1KZALKJ/BXTQ2U4U/",
# "/root/chujiajia/DataSet/test/cine/8/A1KZALKJ/BXTQ1E4U/",
# "/root/chujiajia/DataSet/test/cine/9/A1KZALKJ/JOWNKWZ4/"]
# spathdir = "/root/chujiajia/DataSet/test/IMg/"
# for path in pathlist:
#     name = path.split('/')[6]
#     spath = os.path.join(spathdir,name)
#     if not os.path.exists(spath):
#         os.mkdir(spath)
#     sitk_save_nii_slice_to_pgm(path,spath)

