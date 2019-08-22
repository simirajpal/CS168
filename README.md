# CS168
DSA Image Classification into TICI Score

This project contains multiple notebooks in attempt to solve the problem of classifying patients with acute stroke 
into TICI scores.
The DSA Image sample set includes many patients with acute stroke of varying TICI score (0, 1, 2a, 2b, 3). 
The images are formated as a 3D array, where time is the third axis. Images can have from 15-30 frames in the time
axis.

### resize_and_padding.ipynb
  We first began by resizing the images to be a more manageable size.
  Some of the tests we ran involved padding the images with zero in the time axis since this was a variable length 
  axis.
  Padding the images was not a good idea because each patient is different in how many pictures were taken and the 
  time intervals of imaging.
  
### manual_convolution.ipynb
  We attempted manually convolving the images prior to applying a DNN for classification.
  
### Perform 2D Models on Averaged Data.ipynb
  We averaged each 3D image using 5 images at the center of the time points (this is the phase where the blood fills the arteries).
  We tested CNNs using this dataset and attempted data augmentation to get better results.
  We attempted to use the Inception-v3 CNN in this method.
  We were able to acheive 47% accuracy.
  
### 3D_model.ipynb
  We attempted to apply the CNN on the 3D images with time as the third axis.
  This did not result in better results than the 2D averaged images approach.
  
### There are many ways to improve this model, some methods may include:
  improve image preprocessing
  instead of generating means, try using difference gradients
  
  
