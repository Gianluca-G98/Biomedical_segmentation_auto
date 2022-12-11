# Introduction
**Use Deep Neural Networks for the task of semantic segmentation of microglial cells**  
The project allows the user to quickly segment microglial samples using Deep neural networks. The project is composed of 2 parts. 
1. **Preprocess and predict:** the raw data gets preprocessed and then fed to the neural network of choice
2. **Postprocessing:** the output of the prediction gets post-processed to increase the precision and remove small artefacts
The only scripts the user is supposed to run are the *.ipynb* and *CleanFolders.py*. 

Please note that the input data is supposed to be images in .tiff format with 4 channels, (Z, MN, H, W). 
- **Z**: z-planes or slices
- **MN**: microglia/nuclei channel
- **H**: height
- **W**: width
Having the data in different formats or shapes might cause errors. The automatic sanity check will help you figure out whether everything is fine.

# 1) Preprocess and predict
The *preprocess_and_predict.ipynb* notebook contains all you need to run for this part.
I'll briefly cover the folders content and meaning

- **input_dir**: directory where the images to be preprocessed and predicted are. Put ONLY the images to be predicted in that folder, there must be no other folders, images or files of any kind.
- **out_preprocessed_npy**: directory where the pre-processed images are saved in .npy format (only for the model)
- **out_pred_npy**: directory where the model will save the prediction in .npy format. These are the data that will be used by post-processing, don't touch them!
- **out_pred_png**: directory where the .png prediction images created by the model should go. Grab these if you want to see what the model predicted.
- **model_dir**: directory where is the ".h5" model that makes the prediction
- **tile_size**: the size of the tile used by the network to make the prediction. Too high a value will crash the model. I recommend keeping it at 256 and wanting to try 512.
- **verbose**: is for debugging, leave it at zero


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