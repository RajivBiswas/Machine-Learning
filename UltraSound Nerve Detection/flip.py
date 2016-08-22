import numpy as np
import os
import glob
import cv2
import datetime
import time
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss

def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    #cv2.imshow("Original", img)
    #resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return cv2.resize(img, (img_cols, img_rows))

def get_im_cv2_resized(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    rimg = img.copy()
    rimg = cv2.flip(img, 1)
    #cv2.imshow("Vertical Flip", rimg)
    #resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return cv2.resize(rimg, (img_cols, img_rows))

def load_train(img_rows, img_cols):
    X_train = []
    X_train_id = []
    mask_train = []
    start_time = time.time()

    print('Read train images')
    files = glob.glob("/home/r/NerveSegmentation/train/*[0-9].tif")
    for fl in files:
        flbase = os.path.basename(fl)
        print('fl Path:', fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_train.append(img)
        img = get_im_cv2_resized(fl, img_rows, img_cols)
        X_train.append(img)
        X_train_id.append(flbase[:-4])
        mask_path = "/home/r/NerveSegmentation/train/" + flbase[:-4] + "_mask.tif"
        print('Mask Path:', mask_path)
        mask = get_im_cv2(mask_path, img_rows, img_cols)
        mask_train.append(mask)
        mask = get_im_cv2_resized(mask_path, img_rows, img_cols)
        mask_train.append(mask)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, mask_train, X_train_id

if __name__ == '__main__':
    img_rows, img_cols = 32, 32
    train_data, train_target, train_id = load_train(img_rows, img_cols)
