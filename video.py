import os
import cv2
from ultralytics import YOLO

curr_path = os.getcwd()
img_path = os.path.join(curr_path,'images')
all_files = os.listdir(img_path)
image_files = [file for file in all_files if os.path.isfile(os.path.join(img_path, file))]
# Cargar el modelo
model2 = YOLO(curr_path + '/runs/detect/train2/weights/best.pt')
img_train_path = os.path.join(curr_path, 'images', 'train')
output_path = os.path.join(curr_path, 'images', 'output')  # Directorio para guardar las imágenes procesadas
os.makedirs(output_path, exist_ok=True)  # Crear el directorio si no existe

available_images = os.listdir(img_train_path)
for image_name in available_images:
    img_path = os.path.join(img_train_path, image_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Realizar la predicción
    res = model2(img_path)
    img_with_predictions = res[0].plot()  

    if img_with_predictions is not None:
        # Convertir de BGR a RGB 
        img_with_predictions = cv2.cvtColor(img_with_predictions, cv2.COLOR_BGR2RGB)
        # Guardar la imagen procesada
        cv2.imwrite(os.path.join(output_path, image_name), img_with_predictions)

print("Todas las imágenes procesadas y guardadas.")

import cv2

output_video_path = os.path.join(curr_path, 'output_video.mp4')
frame_size = (1920, 1080)  # Ajustar al tamaño deseado o al tamaño de las imágenes
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Código de compresión para el formato MP4
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, frame_size)  # 30 fps

output_images = sorted(os.listdir(output_path))  # Asegurarse de que las imágenes están en orden correcto
for image_name in output_images:
    img_path = os.path.join(output_path, image_name)
    img = cv2.imread(img_path)
    if img is not None:
        img_resized = cv2.resize(img, frame_size)  # Asegurarse de que la imagen tiene el tamaño del frame
        video_writer.write(img_resized)

video_writer.release()
print("Video creado exitosamente en:", output_video_path)
