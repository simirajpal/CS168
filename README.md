# CS168
DSA Image Classification into TICI Score

This project contains multiple notebooks in attempt to solve the problem of classifying patients with acute stroke 
into TICI scores.\n
The DSA Image sample set includes many patients with acute stroke of varying TICI score (0, 1, 2a, 2b, 3).\n 
The images are formated as a 3D array, where time is the third axis. Images can have from 15-30 frames in the time
axis.\n

### resize_and_padding.ipynb
  We first began by resizing the images to be a more manageable size.\n
  Some of the tests we ran involved padding the images with zero in the time axis since this was a variable length 
  axis.\n
  Padding the images was not a good idea because each patient is different in how many pictures were taken and the 
  time intervals of imaging.\n
  
### manual_convolution.ipynb
  We attempted manually convolving the images prior to applying a DNN for classification.\n
  
### Perform 2D Models on Averaged Data.ipynb
  We averaged each 3D image using 5 images at the center of the time points (this is the phase where the blood fills the arteries).\n
  We tested CNNs using this dataset and attempted data augmentation to get better results.\n
  We attempted to use the Inception-v3 CNN in this method.\n
  We were able to acheive 47% accuracy.\n
  
### 3D_model.ipynb
  We attempted to apply the CNN on the 3D images with time as the third axis.\n
  This did not result in better results than the 2D averaged images approach.\n
  
### There are many ways to improve this model, some methods may include:
  improve image preprocessing\n
  instead of generating means, try using difference gradients\n
  
  
