# ImageSegmentation
Image segmentation for FIB-SEM Cell Microstructure Images using Unet and similarly structured CNNs

Notebooks explain the usage of different aspects of the analysis

See [RunUnet2D.ipynb](https://github.com/aduenner/ImageSegmentation/blob/master/RunUnet2D.ipynb) for how to:
  - Import training data
  - Define a fully paramaterized UNET model
  - **Train UNET model**
  - Generate sub-images from a large grayscale image
  - Visualize predictions on test data
  
  See [PC_Image_Generator.ipynb](https://github.com/aduenner/ImageSegmentation/blob/master/PC_Image_Generator.ipynb) for:
   - Calculate principal components
   - Visualize principal components
   - **Generate "fake" images** based on principal components calculated from real images**

See [3D_Visualization](https://github.com/aduenner/ImageSegmentation/blob/master/3d_visualization.ipynb) for working with a sliced dataset (WIP)
