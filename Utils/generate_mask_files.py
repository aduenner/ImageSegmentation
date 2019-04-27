import os
import glob
import numpy as np
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
from mask_functions import *
import warnings

        
def create_masks(model_inputs):
    
    training_dir = model_inputs['training_directory']
    mask_dir = model_inputs['mask_directory']
    categories = model_inputs['category_labels']
    color_dict = model_inputs['mask_color_dict']
                                                                         
    files = glob.glob(mask_dir+'\*.png')
    generated_files=[]

    for file in files:
        image = imread(file)
        filename = file.split('\\')[-1]

        binary_mask = binarize_mask(image,color_dict,categories)
        mask_image = multichannel_mask_array(binary_mask,categories)

        for idx,category in enumerate(categories):
            file_out_name = filename.strip('.png')+'_'+category+('.png')
            dir_name = os.path.join(training_dir,category)
            
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                print("Directory " , dir_name ,  " Created ")
            
            file_out_path = os.path.join(dir_name,file_out_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not os.path.isfile(file_out_path):
                    imsave(file_out_path,mask_image[:,:,idx])
                
            generated_files.append(file_out_path)
        
    print("Created "+str(len(generated_files))+" mask files in "+training_dir)
    
    return generated_files

def cleanup_masks(generated_files):
    for file in generated_files:
        os.remove(file)