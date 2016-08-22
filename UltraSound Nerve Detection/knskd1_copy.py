# This script loops through ultrasound images in the training set that have non-blank masks,
# and then plots each image, adding the edges of their respective masks in red.
# This should outline the BP nerves in the training images. 
# Chris Hefele, May 2016

IMAGES_TO_SHOW = 1  # configure to taste :)


import numpy as np
import matplotlib.pyplot as plt
import glob, os, os.path
import cv2
from lasagne.updates import nesterov_momentum
from lasagne import layers
from lasagne.updates import adadelta, apply_momentum
from nolearn.lasagne import NeuralNet
from array import *
xHeight = 0
xWidth  = 0
count   = 0

def image_with_mask(img, mask):
    # returns a copy of the image with edges of the mask added in red
    #img_color = grays_to_RGB(img)	
    smpl_pt_l_x = array('I', [])
    smpl_pt_l_y = array('I', [])
    smpl_pt_r_x = array('I', [])
    smpl_pt_r_y = array('I', [])
    img_color = np.zeros([len(img), len(img[0]), 3],dtype=np.uint8)
    global xHeight
    global xWidth
    xHeight = len(img)
    xWidth  = len(img[0])
    img_color.fill(255)
    mask_edges1 = cv2.Canny(mask, 0, 0) > 0
    mask_edges = cv2.Canny(img, 0, 0) > 0  
    img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    #img[mask_edges] = 0
    #print('plotted:', mask)
    for i in range(len(mask)):
    	for j in range(len(mask[0])):
		if mask[i][j] == 0:
			img_color[i, j, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
			img_color[i, j, 1] = 255
			img_color[i, j, 2] = 255



    img_color[mask_edges1, 0] = 0  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges1, 1] = 64
    img_color[mask_edges1, 2] = 0
    global count
    count = 0
    for i in range(len(mask)):
	index = 0
    	for j in range(len(mask[0])):
		if img_color[i, j, 1] == 64:
			smpl_pt_r_x.insert(count, j)
			smpl_pt_r_y.insert(count, i)
			if j > index and index != 0:
				index = 0
			else:
				smpl_pt_l_x.insert(count, j)
				smpl_pt_l_y.insert(count, i)
				index = j
			count = count + 1
			
		
	
    
    return img_color
    #return mask

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
	attr_name = getattr(nn, self.name)
	attr_name = new_value

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    #input_shape=(None, 9216),  # 96x96 input pixels per batch
    input_shape=(None, (xWidth*xHeight)),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    dropout_p=0.3,
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=count,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    #update_rho=0.9,
    #update_epsilon=1e-06,

    regression=True,  # flag to indicate we're dealing with regression problem
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=1000,
    verbose=1,
    )
    #max_epochs=2000,  # we want to train this many epochs
    #verbose=1,
    #)

def fimg_to_fmask(img_path):
    # convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img)) 

def plot_image(img, title=None):
    plt.figure(figsize=(15,20))
    plt.title(title)
    plt.imshow(img)
    plt.show()

def main():

    f_ultrasounds = [img for img in glob.glob("/home/r/NerveSegmentation/train/*.tif") if 'mask' not in img]
    # f_ultrasounds.sort()  
    f_masks       = [fimg_to_fmask(fimg) for fimg in f_ultrasounds]
    
    images_shown = 0 
	
    for f_ultrasound, f_mask in zip(f_ultrasounds, f_masks):

        img  = plt.imread(f_ultrasound)
        mask = plt.imread(f_mask)

        if mask_not_blank(mask):

            # plot_image(grays_to_RGB(img),  title=f_ultrasound)
            # plot_image(grays_to_RGB(mask), title=f_mask)

            f_combined = f_ultrasound + " & " + f_mask
            #img        = image_with_mask(img, mask)
            plot_image(image_with_mask(img, mask), title=f_combined)
            plot_image(img, title=f_combined)
            print('plotted:', f_combined)
            images_shown += 1

        if images_shown >= IMAGES_TO_SHOW:
            break

    df    = []
    MyImg = np.zeros([len(img), len(img[0])],dtype=np.uint8)
    f_ultrasounds = [img for img in glob.glob("/home/r/NerveSegmentation/test/*.tif")]
    for f_ultrasound in zip(f_ultrasounds):
        img  = plt.imread(f_ultrasound)
    	df.append(img)

main()
