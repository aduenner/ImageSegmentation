clear all, close all
addpath('helper-functions')

%% load in images and mturk masks
masks_directory = 'Training/Trial5/masks/'; % enter with / at end
files_to_process = get_filenames(masks_directory);

for ii = 1:numel(files_to_process)
    data(ii) = load_images(files_to_process(ii), masks_directory);
end

                             
%% manually approve
for ii = 1 : numel(data)
    compare_mask_to_image(data(ii));
 
    keystroke = input('Enter 0 to reject image, or 1 to approve: ');
    if keystroke == 1
        data(ii).approval = 'approved';
    elseif keystroke == 0
        data(ii).approval = 'rejected';
    else
        fprintf('Expecting keystroke to be 0 or 1. Rejecting image.');
        data(ii).approval = 'rejected';
    end
     
end

%% export
for ii = 1 : numel(data)
    if strcmp(data(ii).approval, 'approved')
        new_location = 'Training/Approved';
    elseif strcmp(data(ii).approval, 'rejected')
        new_location = 'Training/Rejected';
    end
    
    image_output_filepath = strcat(new_location, ...
                                   '/', ...
                                   data(ii).id, ...
                                   '_image.png');
    mask_output_filepath = strcat(new_location, ...
                                  '/', ...
                                  data(ii).mturk_filename);
                              
    fprintf('Copying file %.0f (%s) to %s\n', ii, data(ii).id, new_location);
    copyfile(data(ii).image_filepath, image_output_filepath);
    copyfile(data(ii).mask_filepath, mask_output_filepath);
                              
end
        