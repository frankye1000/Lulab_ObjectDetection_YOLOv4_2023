import sys
sys.path.append("..")

from models import Yolov4
from tensorflow import keras
from config import yolo_config    # 要記得改!!!!!!
from utils import DataGenerator, read_annotation_lines

import pickle
from glob import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import os
# 使用第一張 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# import random
# seed = 500
# tf.random.set_seed(500)
# np.random.seed(seed)
# random.seed(seed)

#22 1102是沒有10%背
#22 1104是有10%背景

#186 0330是沒有10%背
train_lines = read_annotation_lines('../dataset/txt/20230331_anno_train_lines_0.txt')   
val_lines   = read_annotation_lines('../dataset/txt/20230331_anno_val_lines_0.txt')



FOLDER_PATH     = '../dataset/img/20230331_train_vali_img_sopbox_0'         # image位置
class_name_path = '../class_names/classes.txt'                              # class位置
data_gen_train  = DataGenerator(train_lines, class_name_path, FOLDER_PATH)  
data_gen_vali   = DataGenerator(val_lines, class_name_path, FOLDER_PATH)

model = Yolov4(weight_path=None, class_name_path=class_name_path)


print('訓練集數量= ', len(train_lines),'驗證集數量= ',len(val_lines))


''''''''''''''''''''''''''''''''''''''
day = '20230403'
checkpoint_filepath = '../model/202303/{}_sopbox_0.weights'.format(day)
val_loss_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

## TODO:
val_mAP_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    # save_weights_only=True,
    monitor='auc',
    mode='max',
    save_best_only=True
    )


callbacks = [val_loss_checkpoint_callback, 
            #  val_mAP_checkpoint_callback
            ]
            
epochs    = 2
his = model.fit(train_data_gen=data_gen_train, 
                initial_epoch = 0,
                epochs       = epochs, 
                val_data_gen = data_gen_vali,
                callbacks    = callbacks,
          )


loss     = his.history['loss']
val_loss = his.history['val_loss']



with open('../model/202303/{}_val_loss.list'.format(day),'wb') as f:
    pickle.dump(val_loss,f) 


with open('../model/202303/{}_loss.list'.format(day),'wb') as f:
    pickle.dump(loss,f) 

with open('../model/202303/{}_loss.his'.format(day),'wb') as f:
    pickle.dump(his,f) 