import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from ultralytics import YOLO

#Path de los datasets
curr_path = os.getcwd()

#Por ahora se están usando las imágenes de entrenamiento para hacer las pruebas, sí se utilizan datos diferentes solo es necesario cambiar el path
img_train_path = os.path.join(curr_path,'images','train')
img_val_path = os.path.join(curr_path,'images','val')
img_test_path = os.path.join(curr_path,'images','test')

label_train_path = os.path.join(curr_path,'labels','train')
label_val_path = os.path.join(curr_path,'labels','val')
label_test_path = os.path.join(curr_path,'labels','test')
data_path = ''
img_path = os.path.join(curr_path,'images')
label_path = curr_path+'/labels'
  

results_list = []
model2 = YOLO(curr_path+'/runs/detect/train2/weights/best.pt')  #Cargar modelo entrenado



#Se hacen las predicciones para cada imagen en el dataset de entrenamiento
for img_name in os.listdir(img_train_path):
    img_path = os.path.join(img_train_path, img_name)
    
    # Obtener predicciones para la imagen actual
    results = model2(img_path)
    results=results[0]
    detections=(len(results))
    
     # Agregar el nombre de la imagen y el número de detecciones a la lista
    results_list.append({
        "Image Name": img_name,
        "Detections": detections
    })

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results_list)

# Guardar el DataFrame en un archivo Excel
results_df.to_excel('detection_results.xlsx', index=False)

"----------------------------------------------------------------------------------------------------------------------------------"

#Calcular las secciones con más detecciones de la imagen

# Leer el archivo Excel con los resultados
df = pd.read_excel('detection_results.xlsx')

# Calcular la suma de detecciones para cada grupo de tres imágenes
secciones = []
for i in range(len(df) - 2):
    total_detecciones = df.iloc[i:i+3]['Detections'].sum()
    secciones.append({
        "Start Image": df.iloc[i]['Image Name'],
        "End Image": df.iloc[i+2]['Image Name'],
        "Total Detections": total_detecciones
    })

# Crear un DataFrame de las secciones
secciones_df = pd.DataFrame(secciones)
available_images = os.listdir(os.path.join(curr_path, 'images', 'train'))

# Ordenar las secciones por el número de detecciones de mayor a menor
secciones_df_sorted = secciones_df.sort_values(by='Total Detections', ascending=False)

# Seleccionar las 50 secciones con más detecciones
top_50_secciones = secciones_df_sorted.head(50)

print(top_50_secciones)

# Seleccionar los 5 rangos con más detecciones
top_5_secciones = top_50_secciones.head(5)

for index, row in top_5_secciones.iterrows():
    start_image = row['Start Image']
    # Extraer el índice numérico del nombre de la imagen y preparar para buscar 3 imágenes válidas
    start_index = int(start_image.split('_')[-1].split('.')[0])

    fig, axes =  plt.subplots(1, 6, figsize=(30, 5)) 
    
    found_images = 0
    j = 0

    while found_images < 3:
        current_image_name = f"video_13min_{str(start_index + j).zfill(3)}.jpg"
        j += 1
        
        if current_image_name in available_images:
            img_path = os.path.join(img_train_path, current_image_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Colocar cada imagen y predicción de forma consecutiva
                axes[found_images * 2].imshow(img)
                axes[found_images * 2].set_title(f"Image: {current_image_name}", fontsize=12)
                axes[found_images * 2].axis('off')

                res = model2(img_path)
                res_plotted = res[0].plot()
                axes[found_images * 2 + 1].imshow(res_plotted)
                axes[found_images * 2 + 1].set_title(f"Predictions on {current_image_name}", fontsize=12)
                axes[found_images * 2 + 1].axis('off')
                
                found_images += 1
        
        if j > 15:  # Prevenir un bucle infinito
            break

    plt.tight_layout()
    plt.show()
