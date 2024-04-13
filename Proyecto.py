import numpy as np 
import pandas as pd 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import matplotlib.pyplot as plt
#from tensorflow.keras.applications import InceptionV3

# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras import backend as K
from keras import applications
from keras.utils import plot_model
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil as sh
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output
import os
import shutil
from xml.etree import ElementTree as ET
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import glob

#Path de los datasets
curr_path = os.getcwd()
img_train_path = os.path.join(curr_path,'images','train')
img_val_path = os.path.join(curr_path,'images','val')
img_test_path = os.path.join(curr_path,'images','test')

label_train_path = os.path.join(curr_path,'labels','train')
label_val_path = os.path.join(curr_path,'labels','val')
label_test_path = os.path.join(curr_path,'labels','test')
data_path = ''
img_path = curr_path+'/images'
label_path = curr_path+'/labels'
faceList = os.listdir(img_path)  


#Dividir el dataset en train, val y test
dataSize = len(faceList)
trainSplit = 0.7
valSplit = 0.1
testSplit = 0.2


imgTrainList = faceList[:int(dataSize*trainSplit)]
imgValList = faceList[int(dataSize*trainSplit):int(dataSize*(trainSplit+valSplit))]
imgTestList = faceList[int(dataSize*(trainSplit+valSplit)):]

imgTrainList = os.listdir(img_train_path)
imgValList = os.listdir(img_val_path)
imgTestList = os.listdir(img_test_path)



def_size = 300
space = ' '
new_line = '\n'
text = '.txt'
class_id = 0

#Función para mover archivos
def move_files(data_list, source_path, destination_path):
    i=0
    for file in data_list:
        filepath=os.path.join(source_path, file)
        dest_path=os.path.join(data_path, destination_path)
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        shutil.move(filepath, dest_path)
        i=i+1
    print("Number of files transferred:", i)

#Función para mover imágenes
def move_images(data_list, source_path, destination_path):
    i=0
    
    for file in data_list:
        filepath=os.path.join(source_path, file)
        dest_path=os.path.join(data_path, destination_path)

        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)
        finalimage_path=os.path.join(dest_path, file)
        img_resized=cv2.resize(cv2.imread(filepath), (def_size, def_size))
        cv2.imwrite(finalimage_path, img_resized)
        i=i+1
    print("Number of files transferred:", i)

def change_extension(file):
    basename=os.path.splitext(file)[0]
    filename=basename+text
    return filename

labelTrainList = list(map(change_extension, imgTrainList))
labelValList = list(map(change_extension, imgValList))
labelTestList = list(map(change_extension, imgTestList))

move_images(imgTrainList, img_path, img_train_path)
