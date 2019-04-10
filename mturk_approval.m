clear all, close all
addpath('helper-functions')

%% load in images and mturk masks
masks_directory = 'Training/Trial7/masks/'; % enter with / at end
files_to_process = get_filenames(masks_directory);

%% load in files of peaks that were already processed
fid_approved = fopen('Training/Approved/approved.txt');
approved_ids = textscan(fid_approved, '%[^\n]');
approved_ids = approved_ids{1};
fid_rejected = fopen('Training/Rejected/rejected.txt');
rejected_ids = textscan(fid_rejected, '%[^\n]');
rejected_ids = rejected_ids{1};
fclose(fid_approved);
fclose(fid_rejected);
                          
for ii = 1:numel(files_to_process)
        data(ii) = load_images(files_to_process(ii), masks_directory);
end

   

%% get rid of ones we've already looked at
elements_to_drop = []; 
for ii = 1 : numel(data)
    if any(strcmp(data(ii).mturk_filename, approved_ids))
        fprintf('already approved %s\n', data(ii).mturk_filename)
        elements_to_drop = [elements_to_drop, ii];
    elseif any(strcmp(data(ii).mturk_filename, rejected_ids))
        fprintf('already rejected %s\n', data(ii).mturk_filename);
        elements_to_drop = [elements_to_drop, ii];
    end
        
end

data(elements_to_drop) = []; 

%% manually approve
for ii = 1 : numel(data)
    compare_mask_to_image(data(ii));
 
    keystroke = input(sprintf('(%.0f/%.0f) Enter 0 to reject image, or 1 to approve: ', ...
        ii, numel(data)));
    if keystroke == 1
        data(ii).approval = 'approved';
        txt_file = 'Training/Approved/approved.txt';
    elseif keystroke == 0
        data(ii).approval = 'rejected';
       
    else
        fprintf('Expecting keystroke to be 0 or 1. Rejecting image.\n');
        data(ii).approval = 'rejected';
        txt_file = 'Training/Rejected/rejected.txt';
    end
    
    fid = fopen(txt_file, 'a');
    fprintf(fid, strcat(data(ii).mturk_filename, '\n'));
    fclose(fid);
     
end

%% export
for ii = 1 : numel(data)
    if strcmp(data(ii).approval, 'approved')
        new_location = 'Training/Approved';
        
    elseif strcmp(data(ii).approval, 'rejected')
        new_location = 'Training/Rejected';
        
    end
    

    mask_output_filepath = strcat(new_location, ...
                                  '/', ...
                                  data(ii).mturk_filename);
                              
    fprintf('Copying file %.0f (%s) to %s\n', ii, data(ii).id, new_location);
    copyfile(data(ii).mask_filepath, mask_output_filepath);
    

                              
end
        