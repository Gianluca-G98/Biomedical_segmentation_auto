# Introduction
**Use Deep Neural Networks for the task of semantic segmentation of microglial cells**  
The project allows the user to quickly segment microglial samples using Deep neural networks. The project is composed of 2 parts. 
1. **Preprocess and predict:** the raw data gets preprocessed and then fed to the neural network of choice
2. **Postprocessing:** the output of the prediction gets post-processed to increase the precision and remove small artefacts
The only scripts the user is supposed to run are the *.ipynb* and *CleanFolders.py*.   

*CleanFolders.py* is a simple script to delete the content of all the output folders, leaving the *input_images* and *models* folders untouched. It is not mandatory, although I prefer to empty the output folder this way after each experiment in order to keep the folders clean for the next one.

Please note that the input data is supposed to be images in .tiff format with 4 channels, (Z, MN, H, W). 
- **Z**: z-planes or slices
- **MN**: microglia/nuclei channel
- **H**: height
- **W**: width
Having the data in different formats or shapes might cause errors. The automatic sanity check will help you figure out whether everything is fine.

# 1) Preprocess and predict
The *preprocess_and_predict.ipynb* notebook contains all you need to run for this part.
I'll briefly cover the folders content and meaning.

- **input_dir:** directory where the images to be preprocessed and predicted are. Put ONLY the images to be predicted in that folder, there must be no other folders, images or files of any kind.
- **out_preprocessed_npy:** directory where the pre-processed images are saved in .npy format (only for the model)
- **out_pred_npy:** directory where the model will save the prediction in .npy format. These are the data that will be used by post-processing, don't touch them!
- **out_pred_png:** directory where the .png prediction images created by the model should go. Grab these if you want to see what the model predicted.
- **model_dir:** directory where is the ".h5" model that makes the prediction
- **tile_size:** the size of the tile used by the network to make the prediction. Too high a value will crash the model. I recommend keeping it at 256 and wanting to try 512.
- **verbose:** is for debugging, leave it at zero

# 2) Postprocessing
To post-process your prediction, simply run the *postprocess.ipynb* notebook. The *config_postproc* at the beginning will allow you to choose the parameters to use. I'll briefly explain now the meaning of the included parameters and folders. 

**Parameters:**  
- **m_area_trsh:** minimum area that a region must have in order not to be discarded
- **m_intensity_trsh:** all regions that have an average intensity in the nuclei channel lower than this threshold are discarded
- **round_n_trsh:** nuclei channel threshold, used to obtain the "prediction" mask of nuclei; round to 1 those values greater than the threshold and puts the rest to zero
- **n_area_trsh:** minimum area that the "predicted" nucleus must have, for a given microglia, in order not to be discarded
- **n_erosion:** number of times we want to apply erosion after post-processing. The higher this parameter, the more likely the cells will detach, but at the same time we worsen the morphology. The output image obtained after erosion can be thought of as a "guide" to identifying the central part of the cell

**Folders:**  
- **out_preprocessed_npy:** directory where the preprocessed data in .npy format is located
- **out_pred_npy:** directory where model predictions are found in .npy format
- **out_postproc:** directory where you want post-processed images to go
- **out_postproc_npy:** directory where the post-processed images in .npy format will be placed (they are used for erosion, look at the .png)
- **out_eroded_png:** directory where the eroded images in .png format will be placed 

**IMPORTANT:** the erosion is applied separately from the rest of the postprocessing. It is only used to get a better estimate of the number of cells in the sample. The .png of the eroded images is also saved, but I suggest not using them, since the erosion ruins the morphology. However, if the input image is very noisy and the morphology predicted is not very good in the first place, it might be possible for eroded images to provide at least a better cell separation.

# Citation (BibTeX)
If you use this software, please cite it as below.

@software{Ggragnaniello_microglia_ss2022,  
  author = {Gragnaniello, Gianluca and Brochier, Lorenzo},  
  month = {12},  
  title = {{Microgial semantic segmentation with Deep learning}},  
  url = {https://github.com/Gianluca-G98/Biomedical_segmentation_auto},  
  version = {1.0},  
  year = {2022}  
}
