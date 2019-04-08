function data = get_filenames(masks_directory);
all_files_in_directory = dir(masks_directory);
all_files_in_directory = {all_files_in_directory.name};
mask_filenames = all_files_in_directory(startsWith(all_files_in_directory, 'SemImage'));
data = cell2struct(mask_filenames, 'mturk_filename', 1);
end