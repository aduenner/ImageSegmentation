from skimage.io import imread
import os
import numpy as np
import mask_functions
from skimage.io import imread
import mask_functions
import Augmentor

def default_pipeline(model_inputs):
    p = Augmentor.Pipeline(model_inputs['image_directory'])
    p.ground_truth(model_inputs['ground_truth_directory'])
    p.crop_by_size(width=model_inputs['image_width'],height=model_inputs['image_height'],probability=1, centre=False)
    #p.random_distortion(probability = 0.02, grid_width = 10, grid_height = 10, magnitude = 1.5)
    p.flip_left_right(probability = 0.5)
    p.flip_top_bottom(probability = 0.5)
    p.rotate(1, max_left_rotation = 25, max_right_rotation = 25)
    p.zoom_random(probability = 0.1, percentage_area = 0.9)
    return p

def generate_n_images(pipeline, model_inputs):
    directory_path = model_inputs['training_directory']
    filenames = os.listdir(directory_path)
    
    X_filenames = [os.path.join(directory_path,name) for name in filenames if name.startswith('Approved')]
    Y_filenames = [os.path.join(directory_path,name) for name in filenames if name.startswith('_groundtruth_')]
    
    if model_inputs['overwrite_images'] == True:
        file_list = X_filenames + Y_filenames
        if file_list:
            for file in file_list:
                try:
                    os.unlink(file)
                except Exception as e:
                    print(e)
        
        pipeline.sample(model_inputs['image_count'])
    
    elif len(X_filenames)==len(Y_filenames) and len(X_filenames) == model_inputs['image_count']:
        print("Using existing files found in training directory")
    else:
        pipeline.sample(model_inputs['image_count'])
    filenames = os.listdir(directory_path)
    X_filenames = [os.path.join(directory_path,name) for name in filenames if name.startswith('Approved')]
    Y_filenames = [os.path.join(directory_path,name) for name in filenames if name.startswith('_groundtruth_')]

    return(X_filenames, Y_filenames)

def read_image(fn):
    img = imread(fn, dtype = np.uint8)
    return(img)

def load_generated_images(X_filenames, Y_filenames, model_inputs):
    
    imwidth  = model_inputs['image_width']
    imheight = model_inputs['image_height']
    input_channels = model_inputs['input_channels']
    categories = model_inputs['categories']
    category_labels = model_inputs['category_labels']
    mask_colors = model_inputs['mask_color_dict']
    image_count = model_inputs['image_count']
    
    #Initialize image arrays
    X_images = np.zeros(
        (len(X_filenames),
         imwidth,
         imheight,
         input_channels),dtype='uint8')

    Y_images = np.zeros(
        (len(X_filenames),
         imwidth*imheight,
         model_inputs['categories']),
        dtype='uint8')

    assert len(X_images) == len(Y_images), "number of X images does not match number of Y images"
    
    #get dims so we can attempt to expand if they are too small
    img_input_dims = np.size(read_image(X_filenames[0]).shape)
    img_output_dims = np.size(read_image(Y_filenames[0]).shape)
    

    #Populate image arrays with images from filenames
    for idx,file in enumerate(X_filenames):
        if img_input_dims == 3:
            X_images[idx,:,:,:] = (read_image(file))
        else:
            X_images[idx,:,:] = np.expand_dims(read_image(file),-1)
    
        Y_image = read_image(Y_filenames[idx])
        
        # convert channels of mask image to dictionary of arrays of binary masks
        binary_mask = mask_functions.binarize_mask(Y_image,mask_colors,category_labels)
        
        # "Hot Shot" by making each mask from above a component of the last dimension in a tensor
        mask_image = mask_functions.multichannel_mask_array(
            binary_mask,category_labels).astype('uint8')
        
        # Reshape masks so that each category is a vector
        vector_masks =  mask_image.reshape(imwidth*imheight,categories)        
        
        Y_images[idx,:,:] = vector_masks

    return(X_images, Y_images)

def test_train_split(X_images, Y_images, model_inputs):
    
    train_fraction = 1 - model_inputs['test_split']
    n_total_images = len(X_images)
    n_train_images = int(n_total_images*train_fraction//1)
    train_numbers  = range(0, n_train_images) # no need to shuffle because randomly generated
    test_numbers   = range(n_train_images, n_total_images) 

    X_train = X_images[train_numbers, :, :, :]
    X_test  = X_images[test_numbers, :, :, :]
    Y_train = Y_images[train_numbers, :, :]
    Y_test  = Y_images[test_numbers, :, :]
    
    return X_train, X_test, Y_train, Y_test

def get_data(model_inputs, pipeline=None):
    if pipeline is None:
        pipeline = default_pipeline(model_inputs)
    
    X_filenames, Y_filenames = generate_n_images(pipeline, model_inputs)
    
    X_images, Y_images = load_generated_images(X_filenames, Y_filenames, model_inputs)
    
    X_train, X_test, Y_train, Y_test = test_train_split(X_images, Y_images, model_inputs)
    
    return X_train, X_test, Y_train, Y_test