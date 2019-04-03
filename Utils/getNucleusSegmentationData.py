import os
import sys
import wget
import zipfile
import warnings

import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize


def getData(model_params, local_dir):
    IMG_HEIGHT = model_params['image_height']
    IMG_WIDTH = model_params['image_width']
    IMG_CHANNELS = model_params['input_channels']

    directory = local_dir+"/Data/"
    zipfilepath = directory+"data-science-bowl-2018.zip"
    print("Loking for file " + zipfilepath)

    if not os.path.exists(directory):
        os.makedirs(directory)

    exists = os.path.isfile(zipfilepath)

    if not exists:
        print("File does not exist so we have to download it...")
        url = 'https://www.dropbox.com/s/j3hx2arbcgt9cb9/data-science-bowl-2018.zip?dl=1'
        current_dir = os.getcwd()
        os.chdir(directory)
        wget.download(url)
        os.chdir(current_dir)

    subdirs = [v[0] for v in os.walk(directory+"/data-science-bowl-2018")]
    if len(subdirs) <= 1:
        print("Unzipping")
        with zipfile.ZipFile(zipfilepath, "r") as zip_ref:
            zip_ref.extractall(directory)

    TRAIN_PATH = directory + "data-science-bowl-2018/input/stage1_train/"
    TEST_PATH = directory + "data-science-bowl-2018/input/stage1_test/"

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    # Get train and test IDs
    print("Train Path is " + TRAIN_PATH)
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')

    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img

    print('Done!')
    return X_train, Y_train, X_test


if __name__ == '__main__':
    model_params = {
    "image_height": 128,
      "image_width": 128,
      "input_channels": 3,
    }
    this_path = os.getcwd();
    main_path = this_path.replace('Data','')
    X_train, Y_train, X_test = getData(model_params, main_path)
