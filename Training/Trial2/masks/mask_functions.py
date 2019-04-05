import numpy as np
from skimage.io import imread, imsave

# Colors for original image
def original_color_dict():
    
    Color_Dict = {
        'goblet': np.array([255,127,14, 255]),
        'microvilli': np.array([31,119,180,255]),
        'nucleus': np.array([44, 160, 44, 255]),
        'basement': np.array([214, 39, 40, 255])   
    }
    return Color_Dict

def binarize_mask(image,color_dict):
    '''Separate RGBa image into a dict of binary images with images matching color_dict
    '''
    img = imread(image)
    mask_dict={}
    imwidth,imheight,channels = img.shape
    
    for mask in color_dict.keys():
        mask_data = np.zeros((imwidth,imheight))
        for ch in range(channels):
            mask_data += (img[:,:,ch]==color_dict[mask][ch])
        mask_dict[mask] = (mask_data==4)
    return mask_dict

def colorize_mask_dict(mask_dict,color_dict):
    '''Return dict of binary masks into color image
    '''
    keys = list(mask_dict.keys())
    imwidth, imheight = mask_dict[keys[0]].shape
                         
    num_masks = len(mask_dict.keys())
    img = np.zeros((imwidth, imheight,4),dtype='uint8')
    for ch in range(4):
        for idx, mask in enumerate(mask_dict.keys()):
            img[:,:,ch]+=mask_dict[mask].astype(img.dtype)*color_dict[mask][ch]
    return img
                        
                         
def multichannel_mask_array(mask_dict):
    '''Turns dict of mask arrays into an 3d array of masks with
    same order as keys in dictionary
    '''
    keys = list(mask_dict.keys())
    imwidth, imheight = mask_dict[keys[0]].shape
    num_masks = len(keys)
    img = np.zeros((imwidth,imheight,num_masks))
    for idx,key in enumerate(keys):
        img[:,:,idx] = mask_dict[key]
    return img
                         
