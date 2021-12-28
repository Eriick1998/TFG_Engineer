#Prepare data
import tensorflow as tf
import keras 
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import cv2
import os 

image_size=(200,200)
train_path=r'C:\Users\Erick\Desktop\TFG\Huevos' #Good / #Bad
valid_path=r'C:\Users\Erick\Desktop\TFG\Validation' #Good /  #Bad
test_path=r'C:\Users\Erick\Desktop\TFG\Test'

###################### Funtion Percentatge ##########################################
train=0.4 
validation=0.5
test=0.1
'''
El objetivo principal es poder igualar las imagenes de huevos malos y buenos, esto se hace
para evitar que durante el entrenamiento de la red neuronal sea descompensado y no pueda generalizar 
de manera correcta. En nuestro caso es necessario dado que el dataset esta muy descompensado.
'''
clases=os.listdir(train_path)# //["Buenos","Malos"]
num_images=[]#// Creamos una lista donde almacenamos el numero de imagenes que hay en cada directorio y su nombre
for e in clases:
    total=len(os.listdir(os.path.join(train_path,e)))
    num_images.append([e,total])
    print("[+] Hay un total de {} imagenes {}".format(total,e))
min_valor=min([i[1] for i in num_images])#// Nos interesa saber de la lista "num_images" que directorio contiene el numero de imagenes mas pequeño

for e in clases:
    os.chdir(os.path.join(train_path,e)) #// Cambiamos de directorio 
    total_images=len(os.listdir(os.path.join(train_path,e)))#// Obtenemos la cantidad de imagenes en ese directorio 
    list_images=os.listdir(os.path.join(train_path,e))#// Obtenemos la lista de imagenes en ese directorio
    if total_images>min_valor:#// Comprobamos que en el directorio actual no haya mas imagenes de las minimas, si es asi, igualamos las cantidad de imagenes
        for e in list_images[min_valor:]:
            os.remove(e)#// Eliminamos las imagenes restantes para equilibrar los directorios.
    else:
        pass   
print("[+]Data now is equal.................................")

################### Data Directory ##################################################
'''
Una vez hemos equilibrado los huevos buenos y malos con el mismo numero de imagenes, 
debemos separar estos en las imagenes que queremos que sean de entrenamiento, validacion
y test.

'''       
for e in clases: #// ["Buenos","Malos"]
    os.chdir(os.path.join(train_path,e))
    total_images=len(os.listdir(os.path.join(train_path,e)))
    operation_1 = int(round(total_images * validation))
    operation_2 = int(round(total_images * test))
    total=os.listdir(os.path.join(train_path,e))
    for i in total[0:operation_1]:
        os.rename(os.path.join(train_path,e,i),os.path.join(valid_path,e,i))
    print("[*]Already validation of {}.............................".format(e))
    total=os.listdir(os.path.join(train_path,e))
    for y in total[0:operation_2]:
        os.rename(os.path.join(train_path,e,y),os.path.join(test_path,y))
    print("[*]Already test of {}...................................".format(e)) 

#####################################################################################   

