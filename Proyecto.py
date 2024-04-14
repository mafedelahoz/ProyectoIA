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
img_path = os.path.join(curr_path,'images')
label_path = curr_path+'/labels'
  
#Leer las imágenes
all_files = os.listdir(img_path)
image_files = [file for file in all_files if os.path.isfile(os.path.join(img_path, file))]

#Dividir el dataset en train, val y test
dataSize = len(image_files)
trainSplit = 0.7
valSplit = 0.1
testSplit = 0.2


imgTrainList = image_files[:int(dataSize*trainSplit)]
imgValList = image_files[int(dataSize*trainSplit):int(dataSize*(trainSplit+valSplit))]
imgTestList = image_files[int(dataSize*(trainSplit+valSplit)):]



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
    if not os.path.isdir(destination_path):
        os.makedirs(destination_path)
        
    i = 0
    for file in data_list:
        src_file_path = os.path.join(source_path, file)
        dst_file_path = os.path.join(destination_path, file)
        
        # Lee, redimensiona y guarda la imagen
        img = cv2.imread(src_file_path)
        img_resized = cv2.resize(img, (def_size, def_size))
        cv2.imwrite(dst_file_path, img_resized)
        
        i += 1
    
    print("Número de archivos transferidos:", i)
def change_extension(file):
    basename=os.path.splitext(file)[0]
    filename=basename+text
    return filename

move_images(imgTestList, img_path, img_test_path)
