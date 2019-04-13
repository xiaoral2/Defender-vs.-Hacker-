##C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi -l 5
import numpy as np
from keras import backend as K
K.backend()
import tensorflow as tf
import os
import cv2
from cnn_model_train import loadCNN
import random
from matplotlib import pyplot as plt
import shutil
from Calculation import gradient

my_dict_1={'mucca':0,'pecora':1}#a
my_dict_3={'mucca':0,'scoiattolo':1}#b
my_dict_2={'pecora':0,'scoiattolo':1}#c
dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    path = ROOT_DIR
    img_path=path+'/raw_img/'
    
    def moveFileto(sourceDir,  targetDir):
        shutil.copy(sourceDir,  targetDir)
    if os.path.exists(os.path.join(path, "badfile.txt")):
        os.remove(os.path.join(path, "bug.txt"))
    print("Which server got attacked?(number from 1-3 only)")
    fail=input()
    failed=int(fail)
    print(img_path)
    print("The server " + fail +" got attacked")
    expect=input("please tell me which you want to choose? please type m,s,or p:\n")
    L=['mucca','pecora','scoiattolo']
    image=['img1','img2','img3','img4','img5','img6','img7','img8','img9']
    if expect=="m":
        r_expect=L[0]
    elif expect =="p":
        r_expect=L[1]
    else:
        r_expect = L[2]

    my_object = gradient(L, failed, path, img_path, r_expect, expect)
    photo_path = my_object.nine_matrix()
    my_object.calculation(photo_path)




