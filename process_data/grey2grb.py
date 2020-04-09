import numpy as np 
from PIL import Image

# lb_con = np.array(Image.open("/root/chujiajia/DataSet/DataProcess/lb_con.png"))
lb_con = np.array(Image.open("/root/chujiajia/DataSet/DataProcess/label.png"))
n_lblcon = np.zeros((256,200,3))

for i in range(lb_con.shape[0]):
	for j in range(lb_con.shape[1]):
		if lb_con[i,j] == 0:
			n_lblcon[i,j,:] = [0,0,0]
		if lb_con[i,j] == 60:
			n_lblcon[i,j,:] = [100,100,100]
		if lb_con[i,j] == 120:
			n_lblcon[i,j,:] = [80,255,100]
		if lb_con[i,j] == 180:
			# n_lblcon[i,j,:] = [0,0,255]
			n_lblcon[i,j,:] = [60,120,255]
		if lb_con[i,j] == 240:
			n_lblcon[i,j,:] = [255,0,0]


n_lblcon = Image.fromarray(np.array(n_lblcon,np.uint8))
n_lblcon.save("/root/chujiajia/DataSet/DataProcess/rgb_label.png")


img = np.array(Image.open("/root/chujiajia/DataSet/DataProcess/im_correct.png"))
contour = np.array(Image.open("/root/chujiajia/DataSet/DataProcess/contour.png"))
