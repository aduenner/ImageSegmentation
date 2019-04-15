%% 
clear all, close all
addpath('helper-functions')


%% list approved images
all_files_in_directory = dir('Training/Approved');
all_files_in_directory = {all_files_in_directory.name};
mask_filenames = all_files_in_directory(startsWith(all_files_in_directory, 'SemImage'));
files_to_process = cell2struct(mask_filenames, 'mask_filename', 1);

%% make a list of which files need to be copied
data = [];
for ii = 1 : numel(files_to_process)
    image_data = files_to_process(ii);
    image_data.new_image_filename = image_data.mask_filename;
    image_data.id = strtok(image_data.mask_filename, '_');
    image_data.original_image_path = strcat(...
        'Training/Raw_Images_1024/', ...
        image_data.id, ...
        '.png');
    image_data.new_image_folder = 'Training/Approved_Raw_Images/';
    image_data.new_image_path = strcat(image_data.new_image_folder, ...
                                       image_data.new_image_filename);
    
    data = [data, image_data];
end

%% actually save the images
for ii = 1 : numel(files_to_process)
    fprintf('Copying file %.0f (%s) to %s\n', ii, data(ii).mask_filename, data(ii).new_image_path);
    copyfile(data(ii).original_image_path, data(ii).new_image_path);
end