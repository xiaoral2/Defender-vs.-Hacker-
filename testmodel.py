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

my_dict_1={'mucca':0,'pecora':1}#a
my_dict_3={'mucca':0,'scoiattolo':1}#b
my_dict_2={'pecora':0,'scoiattolo':1}#c
dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

server=[]
def calculation(failed,photo_path):
    server = [val for val in range(1,4)]
    server.remove(failed)
    print(server)
    position=(server[0]+server[1])/2
    print(position)
    if position == 1.5:
        print("hello server 1 & 2, The server 3 will not going to join the mission. Please finsh the job by your self. Good Luck!")
        model_a= loadCNN()
        model_b= loadCNN()
        model_a.load_weights('./model/model_D1.hdf5')
        model_b.load_weights('./model/model_D2.hdf5')

    if position ==2:
        print("hello server 1 & 3,The server 3 will not going to join the mission. Please finsh the job by your self. Good Luck!")
        model_a= loadCNN()
        model_c= loadCNN()
        model_a.load_weights('./model/model_D1.hdf5')
        model_c.load_weights('./model/model_D3.hdf5')

    if position ==2.5:
        print ("hello server 2 & 3,The server 3 will not going to join the mission. Please finsh the job by your self. Good Luck!")
        model_b= loadCNN()
        model_c= loadCNN()
        model_b.load_weights('./model/model_D2.hdf5')
        model_c.load_weights('./model/model_D3.hdf5')

def nine_matrix(r_expect,L,image,img_path):
    L.remove(r_expect)
    path1=img_path+L[0]+'/'
    path2=img_path+L[1]+'/'
    path3=img_path+r_expect+'/'
    result= int(random.uniform(1,4))
    second=int(random.uniform(1,9-result))
    final=9-result-second
    #    print(L)
    img_L0=os.listdir(path1)
    img_L1=os.listdir(path2)
    img_re=os.listdir(path3)
    counter =0

    img_path=[]
    
    for i in random.sample(img_re,len(img_re)):
        if result!=0:
            image[counter]=cv2.imread(path3+str(i))
            img_path.append(path3+str(i))
            plt.subplot(331+counter),plt.imshow(image[counter])
            result=result-1
            counter=counter+1
    for i in random.sample(img_L0,len(img_L0)):
        if second!=0:
            image[counter]=cv2.imread(path1+str(i))
            img_path.append(path1+str(i))
            plt.subplot(331+counter),plt.imshow(image[counter])
            second=second-1
            counter=counter+1
    for i in random.sample(img_L1,len(img_L1)):
        if final!=0:
            image[counter]=cv2.imread(path2+str(i))
            img_path.append(path2+str(i))
            plt.subplot(331+counter),plt.imshow(image[counter])
            final=final-1
            counter=counter+1
    plt.show()
    return result, img_path

# value = np.array([[0] * 9] * 9, dtype=object)

def img_test():
    model_a= loadCNN()
    model_b= loadCNN()
    model_c= loadCNN()
    model_a.load_weights('./model/model_D1.hdf5')
    model_b.load_weights('./model/model_D2.hdf5')
    model_c.load_weights('./model/model_D3.hdf5')
    
    img = cv2.imread('./raw_img/mucca/OIP-zNuTGA5UlMGm-i_m6PHfNQHaE7.jpeg')
    img = cv2.resize(img,(100,100))
    img = np.array(img, dtype = 'f')
    img = img/255.0
    img = img.reshape([-1,100,100,3])
    
    pdt_a = model_a.predict(img)
    pdt_b = model_b.predict(img)
    pdt_c = model_c.predict(img)
    print(pdt_a, pdt_b, pdt_c)


if __name__ == '__main__':
    path = ROOT_DIR+'/'
    img_path=path+'raw_img/'
    
    def moveFileto(sourceDir,  targetDir):
        shutil.copy(sourceDir,  targetDir)
    if os.path.exists(os.path.join(path, "bug.txt")):
        os.remove(os.path.join(path, "bug.txt"))
    print("Which server got attacked?(number from 1-3 only)")
    fail=input()
    failed=int(fail)
    print("The server " + fail +" got attacked")
    print("please tell me which you want to choose? please type m,s,or p")
    expect=input()
    L=['mucca','pecora','scoiattolo']
    image=['img1','img2','img3','img4','img5','img6','img7','img8','img9']
    if expect=="m":
        r_expect=L[0]
    elif expect =="p":
        r_expect=L[1]
    else:
        r_expect=L[2]

    nvalue, photo_path=nine_matrix(r_expect,L,image,img_path)
#print(photo_path)
    calculation(failed,photo_path)
#img_test()
    os.chdir(path)
    
    print("The mission got completed! The there are "+ str(nvalue) +" of the " +str(expect)+" in the picture showed!")
    if nvalue == nvalue:
        file_transfer=open("bug.txt",'w')
        file_transfer.write("hello world")
        file_transfer.close()

