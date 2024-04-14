import numpy as np 
import pandas as pd 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import matplotlib.pyplot as plt
#from tensorflow.keras.applications import InceptionV3
import copy
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

#Parte 1
"""Creación de los directorios de las imágenes y las etiquetas-Desplazamiento de los datos"""
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
  
def change_extension(file_name, new_extension='.txt'):
    """
    Cambia la extensión de un nombre de archivo a una nueva especificada.
    
    Args:
    file_name (str): Nombre del archivo original.
    new_extension (str): Nueva extensión del archivo, incluyendo el punto (por defecto es '.txt').
    
    Returns:
    str: Nombre del archivo con la nueva extensión.
    """
    # Extraer el nombre base sin la extensión
    basename = os.path.splitext(file_name)[0]
    # Devolver el nuevo nombre de archivo con la nueva extensión
    return basename + new_extension

#Leer las imágenes
all_files = os.listdir(img_path)
image_files = [file for file in all_files if os.path.isfile(os.path.join(img_path, file))]

#Leer las etiquetas
all_labels= os.listdir(label_path)
label_files = [file for file in all_labels if os.path.isfile(os.path.join(label_path, file))]

#Dividir el dataset en train, val y test
dataSize = len(image_files)
trainSplit = 0.7
valSplit = 0.1
testSplit = 0.2

# Asumiendo que los nombres de los archivos de imágenes y etiquetas coinciden antes de la extensión
image_files = sorted([file for file in all_files if file.endswith('.jpg')])  # Ajusta la extensión según tus archivos
label_files = sorted([change_extension(file) for file in image_files])

# Emparejar y aleatorizar simultáneamente
data = list(zip(image_files, label_files))
random.shuffle(data)
image_files, label_files = zip(*data)

# Ahora divide en entrenamiento, validación y pruebas
imgTrainList, labelTrainList = zip(*data[:int(dataSize*trainSplit)])
imgValList, labelValList = zip(*data[int(dataSize*trainSplit):int(dataSize*(trainSplit+valSplit))])
imgTestList, labelTestList = zip(*data[int(dataSize*(trainSplit+valSplit)):])



def_size = 300
space = ' '
new_line = '\n'
text = '.txt'
class_id = 0

#Función para mover archivos
def move_files(data_list, source_path, destination_path):
    # Asegúrate de que el directorio de destino existe, si no, créalo
    if not os.path.isdir(destination_path):
        os.makedirs(destination_path)
        
    i = 0
    for file in data_list:
        src_file_path = os.path.join(source_path, file)
        dst_file_path = os.path.join(destination_path, file)
        
        # Mueve el archivo al directorio de destino
        shutil.copy(src_file_path, dst_file_path)
        i += 1
    
    print("Número de archivos transferidos:", i)


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

""""
move_files(labelTrainList, label_path, label_train_path)
move_files(labelValList, label_path, label_val_path)
move_files(labelTestList, label_path, label_test_path)
move_images(imgTrainList, img_path, img_train_path)
move_images(imgValList, img_path, img_val_path)
move_images(imgTestList, img_path, img_test_path)
"""""

#Parte 2



# Generar líneas de configuración
ln_1 = '# Train/val/test sets' + new_line
ln_2 = 'train: ' + "'" + img_train_path + "'" + new_line
ln_3 = 'val: ' + "'" + img_val_path + "'" + new_line
ln_4 = 'test: ' + "'" + img_test_path + "'" + new_line
ln_5 = new_line
ln_6 = '# Classes' + new_line
ln_7 = 'nc: ' + str(5) + new_line  # Actualizando el número de clases a 5
ln_8 = "names: ['Vehiculos', 'Construcciones', 'Vias', 'Rios', 'Mineria']"

config_lines = [ln_1, ln_2, ln_3, ln_4, ln_5, ln_6, ln_7, ln_8]

#Creación del archivo config.yaml
config_path=os.path.join(curr_path, 'config.yaml')
config_path
green = (0,255,0)


def get_bbox_from_label(text_file_path):
    bbox_list=[]
    print(text_file_path)
    with open(text_file_path, "r") as file:
        
        for line in file:
            _,x_centre,y_centre,width,height=line.strip().split(" ")
            x1=(float(x_centre)+(float(width)/2))*def_size
            x0=(float(x_centre)-(float(width)/2))*def_size
            y1=(float(y_centre)+(float(height)/2))*def_size
            y0=(float(y_centre)-(float(height)/2))*def_size

            vertices=np.array([[int(x0), int(y0)], [int(x1), int(y0)],
                               [int(x1),int(y1)], [int(x0),int(y1)]])
#             vertices=vertices.reshape((-1,1,2))
            bbox_list.append(vertices)

    return tuple(bbox_list)

plt.figure(figsize=(30,30))
for i in range(1,8,2):
    k = random.randint(0, len(imgTrainList)-1)
    img_path = os.path.join(img_train_path, imgTrainList[k])
    label_path = os.path.join(label_train_path, labelTrainList[k])
    print(img_train_path,label_train_path)
    bbox = get_bbox_from_label(label_path)
    img = cv2.imread(img_path)
    copy_img = copy.deepcopy(img)
    ax=plt.subplot(4, 2, i)
    plt.imshow(img) # displaying image
    plt.xticks([])
    plt.yticks([])
    cv2.drawContours(copy_img, bbox, -1, green, 2)
    ax=plt.subplot(4, 2, i+1)
    plt.imshow(copy_img) # displaying image with bounding box
    plt.xticks([])
    plt.yticks([])
    
