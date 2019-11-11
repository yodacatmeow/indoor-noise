# Type/position classification of inter-floor noises in residential buildings with a single microphone via supervised learning

(Under constuction...)



This is the implementation code for the paper submitted for ICASSP 2020.  



### Dataset

A convolutional neural networks based type/position classifier was tested against the following datasets to address the generalizability of the classifier.

- [CS-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/CS-APT)
- [BDML-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/BDML-APT)



### Code implementation

- Download the two datasets and locate all audio files to ```audio```  folder.
- Generate a training/validation set and a test set.
  - Open ```dataset.py```
  - Select ```fold_conf``` in ```pp_n```
  - Start a process ```python3 dataset.py```
- Hyperparameter searching via random search



### Results



### Citing