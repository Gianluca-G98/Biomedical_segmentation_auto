import os
import numpy as np
from preprocess import preprocess
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import time
from skimage.morphology import binary_erosion

def postprocess(y_pred, intensity_img, config):
    '''
    y_pred: 2D array, binary annotation
    intensity_img: 3 D array (H, W, Ch), where the first slice (H, W, 0) is microglia and
        the second one is nuclei intensity image; default values are chosen supposing
        intensity_img values are in range [0-1].
    '''

    # check if the prediction and the instensity images have the same shape
    assert y_pred.shape == (intensity_img.shape[0], intensity_img.shape[1])

    # check intensity_img dimensions
    assert len(intensity_img.shape) == 3
    assert intensity_img.shape[2] == 2

    # check if y_pred is binary
    assert (np.unique(y_pred) == [0, 1]).all()

    # check if the nucleus slice is between 0 and 1
    assert (np.min(intensity_img[:, :, 1]) >= 0, np.max(intensity_img[:, :, 1])) == (True, 1)

    # assign different label for each instance
    pred_lab = label(y_pred, connectivity=2) 

    # transform nuceli intensity image in binary image and concatenate it at the end
    # of intensity image
    rounded_n_img = np.where(intensity_img[:, :, 1] >= config['round_n_trsh'], 1, 0)
    intensity_img = np.concatenate((intensity_img, np.expand_dims(rounded_n_img, axis=-1)), axis=-1)

    # create a mask that account for nuclei inside predicted microglia cells
    # Replace Nuceli mask with this newly created mask (n_in_m)
    n_in_m = np.where(pred_lab > 0, 1, 0) * intensity_img[:,:, 2] # M binary mask x N binary mask
    intensity_img[:, :, 2] = n_in_m


    # For area

    # list of objects. each object represents a single instance (predicted cell pixels).
    # Each object has multiple attributes which can be accessed
    props = regionprops(pred_lab, intensity_img)

    # list of labels that represents instances to be deleted from y_pred
    idxs = []

    for obj in props:
        # array of intensities of microglia in the region of 'n_in_m'
        arr = obj.image_intensity[:, :, 0][np.ma.make_mask(obj.image_intensity[:, :, 2])]

        # arr is empty when the respective n_in_m mask is full of False
        if arr.size == 0:
            int_micr = -1 # set to -1 to delete cells when m_intensity_trsh is 0 and rounded nuclei is not present in the region
        else:
            int_micr = np.mean(arr)

        if obj.area < config['m_area_trsh'] or int_micr < config['m_intensity_trsh'] or arr.size < config['n_area_trsh']:
            # appending label to list of labels
            idxs.append(obj.label)

    # transform from list to array
    idxs = np.array(idxs)

    print('Number of instances to be deleted: ',f'{idxs.size}/{len(props)} ({idxs.size / len(props):.5f})')

    # creating the binary mask to then modify y_pred
    # check if element of 'pred_lab' is in 'idxs' and if it is, return 1 in the 
    # same position of the checked element in pred_lab
    mask = np.isin(pred_lab, idxs)

    #compute percentage of predicted 1 that will be turned to 0 w.r.t. total 1 in y_pred
    perc_true = (np.sum(mask)) / np.sum(y_pred)
    print(f'Percentage of pixels deleted among y_pred==1 : {perc_true:.5f}', '\n')

    # creating new prediction image
    new_pred = np.where(mask == 1, 0, y_pred)

    return new_pred


def post_proc_save(config, verbose=0):
    for root, dirs, files in os.walk(config["out_pred_npy"], topdown=False):
        for file in files:
            if file != '.gitkeep':
                start = time.time()
                pred_path = config["out_pred_npy"] + "/" + file
                y_pred = np.load(pred_path)

                fluo_path = config["out_preprocessed_npy"] + '/' + file
                fluo = np.load(fluo_path)

                # The prediction and fluorescence dimensions must match. 
                # Obviously fluorescence is three-dimensional so we take a slice at random
                if y_pred.shape != fluo[:,:,0].shape:
                    print("ATTENZIONE! Le dimensioni di prediction e fluorescenza preprocessata non combaciano.")
                    print("y_pred shape: {}\nfluo shape: {}".format(y_pred.shape, fluo[:,:,0].shape))
                    raise Exception()

                if verbose > 0:
                    print("pred_path: {}\nfluo_path: {}".format(pred_path, fluo_path))

                post_proc_img = postprocess(y_pred, fluo, config)
                plt.imsave(config["out_postproc"]+f'/{file[:-4]}.png', post_proc_img)
                np.save(config["out_postproc_npy"]+f'/{file[:-4]}.npy', post_proc_img)
                print(f'Post processing of {file} saved successfully!! (.npy and .png)')
                end = time.time()
                print("ETA: ", end - start)
                print('----------------------------------')
            else:
                continue


def erode_images(config):
    cellcount_dict = dict()

    # list of regions labels to be deleted
    for root, dirs, files in os.walk(config['out_postproc_npy'], topdown=True):
        for file in files:
            start = time.time()
            if file != '.gitkeep':
                # list of regions labels to be deleted
                idxs = []
                
                path = config['out_postproc_npy'] + '/' + file
                postproc = np.load(path)
                eroded_img = binary_erosion(postproc)
                for i in range(config['n_erosion']-1):
                    eroded_img = binary_erosion(eroded_img)
                labels_eroded = label(eroded_img)
                for region in regionprops(labels_eroded):
                    if region.area < 150: # setting a threshold depending from config thrsh
                        idxs.append(region.label)
                
                # mask of cells to be deleted
                idxs = np.array(idxs)
                mask = np.isin(labels_eroded, idxs)

                # postprocessed labels (without cells with label in idxs)
                labels_postproc = np.where(mask == 1, 0, labels_eroded)
                cellcount_dict[file[:-4]] = len(np.unique(labels_postproc))
                eroded_postproc = np.where(labels_postproc > 0, 1, 0)

                # save image
                plt.imsave(config['out_eroded_png']+ '/' + file[:-4] + '.png', eroded_postproc)
                print(f'Erosion of {file[:-4]} saved successfully!! (.png)')
                end = time.time()
                print("ETA: ", end - start)
            else:
                continue
                 
    # Now we write the cell_count in a separate txt file
    with open("eroded_cell_count.txt", 'w') as f:
        for key, value in cellcount_dict.items():
            f.write('%s:%s\n' % (key, value))
    return cellcount_dict