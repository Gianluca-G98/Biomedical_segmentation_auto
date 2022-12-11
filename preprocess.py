# -*- coding: utf-8 -*-

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_unet_collection.losses import dice, tversky
from skimage import exposure, io

tfk = tf.keras
tfkl = tf.keras.layers

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


# Check to make sure everything is okay
def sanity_check(config):
    ### Checking that directory names do not have the final slash ###
    directory_list = [
        config['input_dir'], 
        config['out_preprocessed_npy'],
        config['out_pred_npy'],
        config['out_pred_png'],
        config['model_dir']]

    for dir in directory_list:
        if dir[-1] == '/':
            raise Exception('Le directories non devono aver uno slash finale')

    ###  I take the images from the input directory, plotting some stuff to check everything is okay.  ###
    for root, dirs, files in os.walk(config['input_dir'], topdown=False):
            for file in files:
                if file != '.gitkeep':
                    # Read the file and run the checks     
                    fluo = io.imread(config['input_dir']+"/"+file)
                    print("Check sul file ", file)
                    print('Images should have shape (Z, 2, H, W), with first microglia and then nuclei')
                    print('The shape of this image is ', fluo.shape)
                    if len(fluo.shape) != 4 or fluo.shape[0] > 1000 or fluo.shape[1] != 2: 
                        print("Wrong shape! Check that the shape is in the shape (Z, 2, H, W).")
                        return "0"

                    print("----------------------------------------------")
                    print("Of the following two images, check that the first is Microglia and the second is nuclei!")
                    plt.imshow(fluo[0,0,:,:])
                    plt.show()
                    plt.imshow(fluo[0,1,:,:])
                    plt.show()

                    # If they press 0 they exit sanity check, otherwise they go to the next image
                    status = input("Press Enter when you want to go to check the next file. \
                        Press zero and after Enter if you want to end the check instead.")
                    if status == "0":
                        return status
            else:
                continue
        
def prediction(img, tile_size, model_dir):
    if len(img.shape) != 3:
        raise Exception('Wrong shape: ', img.shape)

    model = tf.keras.models.load_model(model_dir, custom_objects={'dice': dice})
    y_pred = np.zeros_like(img[:,:,0], dtype=np.uint8)  # Create a ndarray for the prediction
    size_vert = tile_size
    size_oriz = tile_size
    i = 0
    for vert in range(0, img.shape[0], size_vert): # iterate vertically
        for oriz in range(0, img.shape[1], size_oriz): # iterate horizontally
            input_tile = img[vert:vert+size_vert, oriz:oriz+size_oriz, 0:1]  # I take the corresponding portion of norm_X
            input_tile = tf.expand_dims(input_tile, axis=0)  # add batch dim

            # take the prediction without the batch and channel dimension
            tile_predicted = model(input_tile, training=False)[0, :,:,0]
            y_pred[vert:vert+size_vert, oriz:oriz+size_oriz] = tile_predicted  # put prediction in the ndarray
            i += 1
    return y_pred


# Function called by preproc_predict() in the loop to preprocess a single image.
# Returns the image it takes as input but preprocessed.
def preprocess(image_dir, file_name,  verbose=0, ):  # Test image shape: (4, 2, 6546, 7390)
    # Reads the image in the input directory
    fluo = io.imread(image_dir+"/"+file_name)

    if verbose > 0:  # Plot di debug
        plt.imshow(fluo[0,0,:,:])
        plt.show()
        plt.imshow(fluo[0,1,:,:])
        plt.show()

    # I cut the image into multiples of 128.
    # Calculate the number of pixels in height
    n_height = (fluo.shape[2] // 128) * 128
    n_width = (fluo.shape[3] // 128) * 128
    fluo = fluo[:, :, 0:n_height, 0:n_width]

    # Fuse z-channels
    ch1 = np.max(fluo[:,0,:,:], axis=0)
    ch2 = np.max(fluo[:,1,:,:], axis=0)

    # Equalize the image
    ch1 = exposure.equalize_adapthist(ch1).astype("float32")
    ch2 = exposure.equalize_adapthist(ch2).astype("float32")

    if verbose > 1:  # Plot di debug
        print(ch1.shape, ch2.shape)
        plt.imshow(ch1)
        plt.show()
        plt.imshow(ch2)
        plt.show()

    # Stack the two images
    final_fluo = np.dstack((ch1, ch2))

    if verbose > 2:
        print(final_fluo.shape)
        plt.imshow(final_fluo)
        plt.show()

    return final_fluo

# This is the "master" function they call from the Notebook. The first thing it does is to run the function of the 
# sanity check. If you exit the check function because you have checked all the images the function
# proceeds to preprocess and predict all the images in the input foldere. 
# If, on the other hand, you exit sanity check manually (by pressing 0), you are asked if you want to continue with the execution
# of the rest of the program (preprocess and predict) or abort the script.
def preproc_predict(config):  
    status = sanity_check(config)
    if status == '0':
        status = input("Press Enter to continue. Press 0 to stop the program.")
        if status == '0':
            return status

    # Iterate in all files in the input directory   
    for root, dirs, files in os.walk(config['input_dir'], topdown=False):
        for file in files:
            start = time.time()
            if file != '.gitkeep':
                # Preprocessing the input image
                fluo = preprocess(config['input_dir'], file, verbose=config['verbose'])  
                # Save the preprocessed image in npy
                np.save(config['out_preprocessed_npy']+f'/{file[:-4]}.npy', fluo) 
                plt.imsave(config['out_preprocessed_png']+f'/M_{file[:-4]}.png', fluo[:,:,0])
                plt.imsave(config['out_preprocessed_png']+f'/N_{file[:-4]}.png', fluo[:,:,1])
                print("Preprocessing of image {} completed! Starting prediction...".format(file)) # checkme

                # Predict the image just pre-processed
                fluo = prediction(img=fluo, tile_size=config['tile_size'], model_dir=config['model_dir'])
                # Save the newly predicted image both in npy, for subsequent post-processing steps
                # and in png for them to see the images
                np.save(config['out_pred_npy']+f'/{file[:-4]}.npy', fluo)
                plt.imsave(config['out_pred_png']+f'/{file[:-4]}.png', fluo)
                print(f'prediction of {file} saved successfully!! (.npy and .png)')
                end = time.time()
                print("ETA: ", end - start)
                print('----------------------------------')
            else:
                continue