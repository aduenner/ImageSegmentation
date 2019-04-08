function image_data = load_images(image_data, masks_directory)
image_data.id = strtok(image_data.mturk_filename, '_');
image_data.image_filepath = strcat('Training/Raw_Images_1024/', ...
                                 image_data.id, ...
                                 '.png');
image_data.image = imread(image_data.image_filepath);
image_data.mask_filepath = strcat(masks_directory, ...
                            image_data.mturk_filename);
image_data.mask = imread(image_data.mask_filepath);
                           
end