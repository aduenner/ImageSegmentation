function compare_mask_to_image(image_struct)

subplot(2, 2, 1)
imshow(image_struct.image);
title(strcat('Image (', image_struct.id, ')'));



subplot(2, 2, 2)
map = colormap([0 0 0;
            0.202 0.478 0.991;
            0.070 0.745 0.725;
            0.786 0.757 0.159;
            0.977 0.983 0.081]);
imshow(image_struct.mask, map)
title(strcat('MTurk mask (', image_struct.mturk_filename, ')'))

subplot(2, 2, 3)
image_masked = image_struct.image;


if ndims(image_struct.mask) == 2
    image_masked(image_struct.mask == 0) = 0; 
elseif ndims(image_struct.mask) == 3
    image_masked(all(image_struct.mask, 3) == 0) = 0;  
else 
    error('Mask did not have expected dimensions')
end

imshow(image_masked);
   

end