import os

import tensorflow_datasets as tfds
import tensorflow as tf
import json
from PIL import Image
import numpy as np
import pandas as pd
import uuid

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir = 'data')

data_config = {}
data_config['train_len'] = info.splits['train'].num_examples
data_config['test_len'] = info.splits['test'].num_examples
data_config['num_classes'] = info.features['species'].num_classes + 1
data_config['color_map'] = {0:[255, 0, 0], 1:[0,255,0], 2:[0,0,0]}
f = open('data/config.json', 'w')
f.write(json.dumps(data_config))
f.close()

def load_image(datapoint): 
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    input_mask -= 1
    input_mask = tf.reshape(input_mask, (tf.shape(input_mask)[0], tf.shape(input_mask)[1]))
    return input_image, input_mask

train = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image , num_parallel_calls=tf.data.AUTOTUNE)

def create_mask_img(mask, color_map):
    img = np.zeros((mask.shape[0], mask.shape[1], 3), np.int64)
    for i in range(len(color_map.keys())):
        color_mask = mask == i
        img[color_mask] = color_map[i]
    return img


all_ids = []
for i, (img, mask) in enumerate(train):
    print('Image: {} / {}   '.format(i+1, data_config['train_len']), end = '\r')
    img_id = uuid.uuid1().hex
    img = Image.fromarray(img.numpy())
    mask = create_mask_img(mask.numpy().astype(np.int64), data_config['color_map'])
    mask = Image.fromarray(mask.astype(np.uint8))
    img_path = os.path.join('data', 'train', '{}_img.png'.format(img_id))
    mask_path = os.path.join('data', 'train', '{}_mask.png'.format(img_id))
    img.save(img_path)
    mask.save(mask_path)
    all_ids.append(img_id)

df = pd.DataFrame()
df['image_ids'] = all_ids
df.to_csv('data/train.csv', index = False)

print()

all_ids = []
for i, (img, mask) in enumerate(test):
    print('Image: {} / {}   '.format(i+1, data_config['test_len']), end = '\r')
    img_id = uuid.uuid1().hex
    img = Image.fromarray(img.numpy())
    mask = create_mask_img(mask.numpy().astype(np.int64), data_config['color_map'])
    mask = Image.fromarray(mask.astype(np.uint8))
    img_path = os.path.join('data', 'test', '{}_img.png'.format(img_id))
    mask_path = os.path.join('data', 'test', '{}_mask.png'.format(img_id))
    img.save(img_path)
    mask.save(mask_path)
    all_ids.append(img_id)

print()

df = pd.DataFrame()
df['image_ids'] = all_ids
df.to_csv('data/test.csv', index = False)


    

    
    
    
    
    

    
    
    
    