import Augmentor
import glob
from natsort import natsorted
import random
import os
import numpy as np
from PIL import Image
from skimage import img_as_float

def get_pipeline(model_inputs):
    image_root_path = model_inputs['image_directory']
    mask_root_path = model_inputs['training_directory']
    categories = model_inputs['category_labels']
    image_list_names=['original']
    image_list = [natsorted(glob.glob(image_root_path+'/*.png'))]
    print("using "+ str(len(image_list[0]))+" images")
    print("with "+ str(len(categories))+" mask categories")
    
    for category in categories:
        search_string = os.path.join(mask_root_path,category+ '/*.png')
        image_list_names.append(category)
        this_image_list = natsorted(glob.glob(search_string))
        image_list.append(this_image_list)

    collated_images_and_masks = list(zip(*image_list))

    images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]

    p = Augmentor.DataPipeline(images)
   
    p.resize(width=model_inputs['image_width'],height=model_inputs['image_height'], probability=1)
    p.flip_left_right(probability = 0.5)
    p.flip_top_bottom(probability = 0.5)
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.zoom_random(probability = 0.2, percentage_area = 0.95)
    
    return p

def multi_generator(pipeline_container, batch_size):
    while True:
        X_images = []
        Y_images = []
        for i in range(batch_size):
            image = pipeline_container.sample(1)[0]
            X_images.append(np.expand_dims(image[0],-1))
            
            num_masks = len(image)-1
            Y_image = np.zeros((256*256,num_masks),dtype='uint8')
            for j in range(num_masks):
                mask = np.expand_dims(image[j+1],-1)
                Y_image[:,j] = mask.reshape(256*256)
            Y_images.append(Y_image)
        
        X_images = np.asarray(X_images)
        X_images = X_images.astype('float32')
        X_images = X_images/np.max(X_images)
        
        
        Y_images = np.asarray(Y_images)
        Y_images = Y_images.astype('float32')
        Y_images = Y_images/np.max(Y_images)
        
        yield (X_images,Y_images)
        
def single_generator(pipeline_container, batch_size):
    while True:
        X_images = []
        Y_images = []
        for i in range(batch_size):
            image = pipeline_container.sample(1)[0]
            X_images.append(np.expand_dims(image[0],-1))
            Y_images.append(np.expand_dims(image[1],-1))
        
        X_images = img_as_float(X_images).astype('float32')
        
        Y_images = np.greater(Y_images,0).astype('float32')
        
        yield (X_images,Y_images)
        
