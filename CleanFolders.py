# Delete automatically all the content inside output folders.
# It DOESN'T delete input images, models or the eroded_cell_count.txt

import os, shutil

folders_list = ['out_eroded_png', 'out_postproc_npy', 'out_postproc_png',
 'out_pred_npy', 'out_pred_png', 'out_preprocessed_npy', 'out_preprocessed_png']

for folder in folders_list:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename != '.gitkeep':  # Check if the selected file is different from .gitkeep
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:  # If it is .gitkeep go to the next file
            continue