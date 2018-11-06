#-*- coding:utf-8 -*-
import cv2
import os
import glob
import numpy as np
import scipy 
import string
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot  as plt
#%matplotlib inline
class DataLoader(object):
    def __init__(self,imgpath):
        self.imgpath = imgpath
        self.alphaDict = {v:idx for idx,v in enumerate(string.ascii_uppercase) }
        self.remapDict = {v:k for k ,v in self.alphaDict.items()}
        # 读取数据
        self.imglist = os.listdir(imgpath)
        self.labellist = [ [ self.alphaDict[v.upper()] for v in i.split('.')[0]] for i in self.imglist]
        
        # 切分训练数据和测试集
        self.idxlist = range(len(self.imglist))
        self.trainset_idx , self.testset_idx = train_test_split(self.idxlist,test_size=0.05)
        
        
        
    def generate(self,shuffle = True,batchSize = 10,trainMode = True):
        if trainMode == True:
            # 写一个yield 循环
            while(True):
                np.random.shuffle(self.trainset_idx)
                for i in range(0,len(self.trainset_idx),batchSize):
                    if len(self.trainset_idx[i:i+batchSize]) == batchSize:
                        imgData = [ cv2.imread( os.path.join(self.imgpath, self.imglist[idx]) )for  idx in self.trainset_idx[i:i+batchSize] ]
                        yield imgData , [self.labellist[idx] for  idx in self.trainset_idx[i:i+batchSize] ] 
        else:
            # 写一个yield 循环
            while(True):
                np.random.shuffle(self.testset_idx)
                for i in range(0,len(self.testset_idx),batchSize):
                    if len(self.testset_idx[i:i+batchSize]) == batchSize:
                        imgData = [ cv2.imread( os.path.join(self.imgpath, self.imglist[idx]) )for  idx in self.testset_idx[i:i+batchSize] ]
                        yield imgData , [self.labellist[idx] for  idx in self.testset_idx[i:i+batchSize] ]
                    
#dataloader = DataLoader('./sample/sample/')
#for  g in dataloader.generate(trainMode=False):
#    plt.imshow(g[0][0])
#    break