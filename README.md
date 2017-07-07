# C3D_BICLSTM

## Prerequisites

1) Tensorflow-0.11 <br/>
2) Tensorlayer (commit ba30379f1b86f930d6e86e1c8db49cbd2d9aa314) <br/> 
   git clone https://github.com/zsdonghao/tensorlayer.git <br/>
   git checkout ba30379f1b86f930d6e86e1c8db49cbd2d9aa314 <br/>
#### The original Tensorlayer does not support the convolutional LSTM, so the tensorlayer/layers.py needs to be replaced with tensorlayer-layers.py. <br/> <br/>
   
## Get the pretrained models
The pretrained models used in training processes can be obtained in the link: https://pan.baidu.com/s/1slc2DMd Password: sty6. <br/>

## How to use the code
### Prepare the data
1) Convert each video files into images using extract_frames.sh in the dataset_splits/video2image.tar.gz. Before running extract_frames.sh, you should change the ROOTDIR in extract_frames.sh, so that IsoGD_phase_1 and IsoGD_phase_2 do exist under $ROOTDIR.
2) Replace the path "/ssd/dataset" in the files under "dataset_splits" with the path "$ROOTDIR"
3) run check_files.py to make sure all necessary image files do exist

### Training Stage
1) Use training_*.py to finetune the networks for different modalities. Please change os.environ['CUDA_VISIBLE_DEVICES'] according to your workstation. <br/>
2) You can run the three training_*.py on three TITAN X GPUs simultaneously. <br/>
### Valid Stage
1) Use svmtrte_isogr_valid.py to train SVM and validate SVM using valid dataset. <br/>
### Testing Stage
1) Use svmtrte_isogr_test.py to train SVM and test SVM using testing dataset. <br/>
#### svmtrte_isogr_test.py reports the final test result. <br/>

## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```
