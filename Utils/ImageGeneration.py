from skimage.io import imread
import os
import numpy as np

def generate_n_images(n, pipeline):
    pipeline.sample(n) 
    filenames = os.listdir(os.path.join(os.getcwd(), 'Training', 'Approved_Raw_Images', 'output'))
    X_filenames = [name for name in filenames if name.startswith('Approved')]
    Y_filenames = [name for name in filenames if name.startswith('_groundtruth_')]
    return(X_filenames, Y_filenames)

def read_image(fn):
    from skimage.io import imread
    path_to_image = os.path.join(os.getcwd(), 'Training', 'Approved_Raw_Images', 'output', fn)
    img = imread(path_to_image, dtype = np.uint8)
    return(img)

def load_generated_images(X_filenames, Y_filenames):
    X_images = map(lambda fn: read_image(fn), X_filenames)
    X_images = np.array(list(X_images))
    X_images = np.expand_dims(X_images, 3)

    Y_images = map(lambda fn: read_image(fn), Y_filenames)
    Y_images = np.array(list(Y_images))
    
    return(X_images, Y_images)

def test_train_split(X_images, Y_images, train_fraction):
    n_total_images = len(X_images)
    n_train_images = np.uint8(np.round(n_total_images*train_fraction, decimals = 0))
    train_numbers = range(0, n_train_images) # no need to shuffle because randomly generated
    test_numbers = range(n_train_images, n_total_images) 

    X_train = X_images[train_numbers, :, :, :]
    X_test = X_images[test_numbers, :, :, :]
    Y_train = Y_images[train_numbers, :, :, :]
    Y_test = Y_images[test_numbers, :, :, :]
    
    return(X_train, X_test, Y_train, Y_test)