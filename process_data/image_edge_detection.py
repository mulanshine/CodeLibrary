from skimage.filters import scharr, gaussian, hessian, prewitt, roberts, sobel
import os
from PIL import Image
import numpy as np 


path = "/root/chujiajia/Dataset/COVID19/train/images/17/076.png"
image = Image.open(path).convert('L')
image.save(os.path.join("/root/chujiajia/Dataset/Process/test/im_17_076.png"))
image = np.array(image)
image = image / 255.0
image = (image - image.mean()) / (image.std() + 1e-7)
ratio = 0.75
image = gaussian(image,sigma=3)
contour = prewitt(image)
contour = (contour - contour.min())/(contour.max()-contour.min()+1e-11)
mid_v = np.sort(contour.reshape(-1))
value = mid_v[int(ratio*len(mid_v))]
# value = 0.2
contour[contour < value] = 0
contour[contour >= value] = 1

cont = contour * 255
# contour[contour < 1e-3] = 0
# contour = 1 - contour
# cont = contour / contour.max() * 255

cont = Image.fromarray(np.array(cont,np.uint8)).convert('L')
cont.save(os.path.join("/root/chujiajia/Dataset/Process/test/con_17_076.png"))


# path = "/root/XieHe_DataSet/npydata/kuoda/P2976721_TAN_YANG_BO/imgs/P2976721_TAN_YANG_BO_CINE_segmented_SAX_b2.npy"
# im = np.load(path)
# # im1 = im[:,:,21]
# image = np.array(im[:,:,22]*255,np.uint8)
# image = image / 255.0
# image = (image - image.mean()) / (image.std() + 1e-7)
# ratio = 0.7
# # image = gaussian(image,sigma=1)
# contour = scharr(image)
# contour = (contour - contour.min())/(contour.max()-contour.min()+1e-11)
# mid_v = np.sort(contour.reshape(-1))
# value = mid_v[int(ratio*len(mid_v))]
# value = 0.07
# contour[contour < value] = 0
# contour[contour >= value] = 1
# cont = contour * 255
# cont = Image.fromarray(np.array(cont,np.uint8)).convert('L')
# cont.save(os.path.join("/root/chujiajia/DataSet/DataProcess/img/cardiac2_contour.png"))
