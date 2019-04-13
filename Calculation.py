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

my_dict = {0:'mucca', 1:'pecora', 2:'scoiattolo'}
my_dict_1={'mucca0':0,'mucca1':0,'pecora0':1,'pecora1':1,'scoiattolo0':2,'scoiattolo1':2}
my_dict_2={'mucca0':0,'mucca2':0,'pecora0':1,'pecora2':1,'scoiattolo0':2,'scoiattolo2':2}
my_dict_3={'mucca1':0,'mucca2':0,'pecora1':1,'pecora2':1,'scoiattolo1':2,'scoiattolo2':2}
'''
my_dict_1={'mucca':0,'pecora':1}#a
my_dict_3={'mucca':0,'scoiattolo':1}#b
my_dict_2={'pecora':0,'scoiattolo':1}#c
'''
dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class gradient(object):
    def __init__(self, L, failed, path, img_path, r_expect,expect):
        self.L = L
        self.failed = failed
        self.path = path
        self.img_path = img_path
        self.r_expect = r_expect
        self.expect=expect
        self.nvalue = -3
    # print(self.img_path)
        #print(self.failed, self.expect, self.path,self.img_path)

    def calculation(self,photo_path):
        server = [val for val in range(1,4)]
        server.remove(self.failed)
        #print(server)
        position = (server[0]+server[1])/2
        #print(position)
        if position == 1.5:
            print("hello server 1 & 2, The server 3 will not going to join the mission. Please finsh the job by your self. Good Luck!")
            model_a = loadCNN()
            model_b = loadCNN()
            model_a.load_weights('./model/model_D1.hdf5')
            model_b.load_weights('./model/model_D2.hdf5')
            counter=0
            L=[]
            for path in photo_path:
                img=cv2.imread(path)
                img = cv2.resize(img,(100,100))
                img = np.array(img, dtype = 'f')
                img = img/255.0
                img = img.reshape([-1,100,100,3])
                pdta=model_a.predict(img).tolist()[0]
                pdtb=model_b.predict(img).tolist()[0]
                pdt_a=[pdta[0]*0.5,pdta[1]]
                pdt_b=[pdtb[0]*0.5,pdtb[1]]
                pdt = [pdt_a[0]+pdt_b[0],pdt_a[1],pdt_b[1]]
                index=(np.argmax(pdt))
                if my_dict[index] ==self.r_expect:
                    counter+=1
        if position == 2:
            print("hello server 1 & 3,The server 3 will not going to join the mission. Please finsh the job by your self. Good Luck!")
            model_a = loadCNN()
            model_c = loadCNN()
            model_a.load_weights('./model/model_D1.hdf5')
            model_c.load_weights('./model/model_D3.hdf5')
            counter=0
            L=[]
            for path in photo_path:
                img=cv2.imread(path)
                img = cv2.resize(img,(100,100))
                img = np.array(img, dtype = 'f')
                img = img/255.0
                img = img.reshape([-1,100,100,3])
                pdta=model_a.predict(img).tolist()[0]
                pdtc=model_c.predict(img).tolist()[0]
                pdt_a=[pdta[0]*0.5,pdta[1]]
                pdt_c=[pdtc[0],pdtc[1]*(-1)]
                pdt = [2*pdt_a[0]-pdt_c[0],2*pdt_a[1]+pdt_c[0],pdt_c[1]]
                index=(np.argmax(pdt))
                if my_dict[index] ==self.r_expect:
                    counter+=1
        if position == 2.5:
            print ("hello server 2 & 3,The server 3 will not going to join the mission. Please finsh the job by your self. Good Luck!")
            model_b = loadCNN()
            model_c = loadCNN()
            model_b.load_weights('./model/model_D2.hdf5')
            model_c.load_weights('./model/model_D3.hdf5')
            counter=0
            L=[]
            for path in photo_path:
                img=cv2.imread(path)
                img = cv2.resize(img,(100,100))
                img = np.array(img, dtype = 'f')
                img = img/255.0
                img = img.reshape([-1,100,100,3])
                pdtb=model_b.predict(img).tolist()[0]
                pdtc=model_c.predict(img).tolist()[0]
                pdt_b=[pdtb[0]*0.5,pdtb[1]]
                pdt_c==[pdtc[0],pdtc[1]*(-1)]
                pdt = [2*pdt_b[0],pdt_c[0],2*pdt_b[1]+pdt_c[1]]
                index=(np.argmax(pdt))
                if my_dict[index] ==self.r_expect:
                    counter+=1
        
        os.chdir(self.path)
        print("The mission got completed! The there are "+ str(self.nvalue) +" of the " +str(self.r_expect)+" in the picture showed!")
            #if counter == self.nvalue:
        file_transfer = open("badfile.txt",'w')
        file_transfer.write("hello world")
        file_transfer.close()

        ##,r_expect,L,image,img_path
    def nine_matrix(self):
        self.L.remove(self.r_expect)
        
        path1=self.img_path + self.L[0]+'/'
        path2=self.img_path + self.L[1]+'/'
        path3=self.img_path+self.r_expect+'/'
        result= int(random.uniform(1,4))
        second=int(random.uniform(1,9-result))
        final=9-result-second
        #    print(L)
        img_L0=os.listdir(path1)
        img_L1=os.listdir(path2)
        img_re=os.listdir(path3)
        counter =0
        
        res_path=[]
        image=['img1','img2','img3','img4','img5','img6','img7','img8','img9']
        print(result)
        self.nvalue=result
        for i in random.sample(img_re,len(img_re)):
            if result!=0:
                image[counter]=cv2.imread(path3+str(i))
                res_path.append(path3+str(i))
                plt.subplot(331+counter),plt.imshow(image[counter])
                result=result-1
                counter=counter+1
        for i in random.sample(img_L0,len(img_L0)):
            if second!=0:
                image[counter]=cv2.imread(path1+str(i))
                res_path.append(path1+str(i))
                plt.subplot(331+counter),plt.imshow(image[counter])
                second=second-1
                counter=counter+1
        for i in random.sample(img_L1,len(img_L1)):
            if final!=0:
                image[counter]=cv2.imread(path2+str(i))
                res_path.append(path2+str(i))
                plt.subplot(331+counter),plt.imshow(image[counter])
                final=final-1
                counter=counter+1
        plt.show()
        return res_path

