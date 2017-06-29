# C3D_BICLSTM

## Prerequisites

1) Tensorflow-0.11 <br/>
2) Tensorlayer (commit ba30379f1b86f930d6e86e1c8db49cbd2d9aa314) <br/> 
   git clone https://github.com/zsdonghao/tensorlayer.git <br/>
   git checkout ba30379f1b86f930d6e86e1c8db49cbd2d9aa314 <br/>
#### The original Tensorlayer does not support the convolutional LSTM, so the tensorlayer/layers.py needs to be replaced with tensorlayer-layers.py. <br/> <br/>
   
## Get the pretrained models
The pretrained models used in training processes can be obtained on the link: https://pan.baidu.com/s/1slc2DMd Password: sty6. <br/>

## How to use the code
### Prepare the data
1) Convert each video files into images
2) Compute the optical flow and save each optical flow frame as one image which is named as "%06d.jpg"
3) If images(RGB, Depth, OpticalFlow) have been prepared and the names are not like "%06d.jpg", you can change the inputs.py.
4) Assign the 'model_prefix' as the path in which you want to store the models in 'training_isogr_*.py'/'testing_isogr_*.py'/'svmtrte_isogr_*.py'.

### Training Stage
1) Use training_*.py to finetune the networks for different modalities. Please replace the paths in the codes with your paths first. <br/>
2) Stop training after 10 epoches are completed.<br/>
### Valid Stage
1) Use testing_isogr_valid.py to validate the networks. Please replace the paths in the codes with your paths first. <br/>
2) Use svmtrte_isogr_valid.py to train SVM and validate SVM using valid dataset. Please replace the paths in the codes with your paths first. <br/>
### Testing Stage
1) Use testing_isogr_test.py to validate the networks. Please replace the paths in the codes with your paths first. <br/>
2) Use svmtrte_isogr_test.py to train SVM and test SVM using testing dataset. Please replace the paths in the codes with your paths first. <br/>


## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```
