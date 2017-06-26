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
Use training_*.py to finetune the networks for different modalities. Please replace the paths in the codes with your paths first. <br/>
Use testing_isogr_valid.py to validate the networks. Please replace the paths in the codes with your paths first. <br/>
Use testing_isogr_test.py to validate the networks. Please replace the paths in the codes with your paths first. <br/>
Use svmtrte_isogr_valid.py to train SVM and validate SVM using valid dataset. Please replace the paths in the codes with your paths first. <br/>
Use svmtrte_isogr_test.py to train SVM and test SVM using testing dataset. Please replace the paths in the codes with your paths first. <br/>


## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```
