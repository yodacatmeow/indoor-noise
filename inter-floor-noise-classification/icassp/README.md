# Type/position classification of inter-floor noises in residential buildings with a single microphone via supervised learning

(Under constuction...)



This is the implementation code for the paper submitted for ICASSP 2020.  



### Dataset

A convolutional neural networks based type/position classifier was tested against the following datasets to address the generalizability of the classifier.

- [CS-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/CS-APT)
- [BDML-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/BDML-APT)



### Code

- Download the two datasets ([CS-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/CS-APT) and [BDML-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/BDML-APT)) and locate all audio files in ```audio```  folder.
- Download the [weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) of VGG16 trained on ImageNet
- Convert the audio files to a training/validation set and a test set.
  - Open ```dataset.py```
  - Write ```fold_conf``` and ```labels``` in ```pp_n``` dictionary
  - Start a process ```python3 dataset.py```
- Hyperparameter searching via random search



### Results



### Citing