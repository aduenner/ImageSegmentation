import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread


def generate_subimages(img_path, sub_size, overlap):
    '''Produces an array of subimages from a large grayscale image
    
    Parameters
    ----------
    img_path : str, full path to image to be segmented ('/path/to/img.png')
    sub_size : int, Width and height of subimage in pixels (subimages are square)
    overlap  : int, Overlap between adjacent subimages in pixels
    '''
    
    #Read image
    input_image = imread(img_path)
    #Convert to Grayscale if color
    if input_image.shape[2]>1:
        input_image = rgb2gray(input_image)
        
    # Scale image and convert to uint8
    # input_image = ((input_image-input_image.min())*255//input_image.max()).astype(np.uint8)
    input_image = ((input_image*255)//1).astype(np.uint8)
    print(input_image.shape)
    width, height = input_image.shape

    numrows=(width-overlap)//sub_size
    numcols=(height-overlap)//sub_size
    
    sub_image_count = numrows*numcols;
    
    #Initialize array of sub_images
    sub_images = np.zeros((sub_image_count,sub_size,sub_size,1),dtype=np.uint8);
    #Keep track of sub image index being processed
    sub_count = 0
    
    #Add sub)images to array
    for row in range(numrows):
        for col in range(numcols):
            startrow = sub_size*row-overlap
            startcol = sub_size*col-overlap
            
            if row == 0:
                startrow = sub_size*row
            
            if col == 0:
                startcol = sub_size*col
            
            endrow = startrow + sub_size
            endcol = startcol + sub_size
            
            sub_image = np.expand_dims(input_image[startrow:endrow,startcol:endcol],axis=-1)
            
            sub_images[sub_count] = sub_image
            sub_count += 1
    
    return sub_images