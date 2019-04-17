import numpy as np
from skimage.io import imread, imsave

# Colors for original image
def original_color_dict():
    
    Color_Dict = {
        'background': np.array([0, 0, 0, 0]),
        'goblet': np.array([255,127,14, 255]),
        'microvilli': np.array([31,110,180,255]),
        'nucleus': np.array([44, 160, 44, 255]),
        'basement': np.array([214, 39, 40, 255])   
    }
    return Color_Dict

def binarize_mask(img,color_dict,mask_list):
    '''Separate RGBa image into a dict of binary images with images matching color_dict
    '''
    mask_dict={}
    imwidth,imheight,channels = img.shape
    
    combined_masks = np.zeros((imwidth,imheight),dtype='uint8')
    
    gen = (mask for mask in color_dict.keys() if mask in mask_list and mask not in ['background'])
    
    for idx,mask in enumerate(gen):
        mask_data = np.zeros((imwidth,imheight),dtype='uint8')
        for ch in range(3):
            mask_data += ((img[:,:,ch]==color_dict[mask][ch])*1).astype('uint8')
            combined_masks += ((img[:,:,ch]==color_dict[mask][ch])*1).astype('uint8')
        mask_dict[mask] = ((mask_data==3)*255).astype('uint8')
    
    mask_dict['background'] = ((combined_masks<3)*255).astype('uint8')
        
    return mask_dict

def colorize_mask_dict(mask_dict,color_dict,mask_list):
    '''Return dict of binary masks into color image
    '''
    imwidth, imheight = mask_dict[keys[0]].shape      
    num_masks = len(mask_list)
    img = np.zeros((imwidth, imheight,3),dtype='uint8')
    
    for ch in range(3):
        for idx, mask in enumerate(mask_list):
            img[:,:,ch]+=mask_dict[mask].astype(img.dtype)*color_dict[mask][ch]
    return img
                        
                         
def multichannel_mask_array(mask_dict,mask_list):
    '''Turns dict of mask arrays into an 3d array of masks with
    same order as keys in dictionary
    '''
    imwidth, imheight = mask_dict[mask_list[0]].shape
    num_masks = len(mask_list)
    img = np.zeros((imwidth,imheight,num_masks),dtype='uint8')
    for idx,mask in enumerate(mask_list):
        img[:,:,idx] = (mask_dict[mask]).astype('uint8')
    return img


def integer_encode(img_dict):
    # Input image mask dictionary and output categories with integer pixels corresponding to categories
    for idx,key in enumerate(list(img_dict.keys())):
        if idx == 0:
            img_out = np.zeros((img_dict[key].shape),dtype='uint8')
        current_mask = (img_dict[key]*idx).astype('uint8')
        img_out += current_mask
    return img_out
        
    
    
                         
