# Type/position classification of inter-floor noises in residential buildings with a single microphone via supervised learning

This is the implementation code for the paper submitted for 28th European Signal Processing Conference (EUSIPCO 2020).  



### Dataset

A convolutional neural networks based type/position classifier was tested against the following datasets to address the generalizability of the classifier.

- [BDML-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/BDML-APT) (APT1)
- [CS-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/CS-APT) (APT2)



### Code implementation

- Download the two datasets ([CS-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/CS-APT) and [BDML-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/BDML-APT)) and locate all audio files in ```audio```  folder.
- Download the [weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) of VGG16 trained on ImageNet and save as ```vgg16_imagenet_1000.npz```
- Convert the audio files to a training/validation set and a test set.
  - Open ```dataset.py```
  - Write ```fold_conf``` and ```labels``` in dictionary ```pp_n```
  - Start a process ```python3 dataset.py```
- Hyperparameter optimization via random search
  - Open ```cross-valid.py``` 
  - Set the parameters in dictionary ```pp``` as follows
    - ```saver = False```
    - ```is_optimal_h = False```
    - ```cnt_max_rand_search = 50```
    - Set ```labels``` to one of ```{'type', 'floor', 'position'}``` 
    - ```n_epoch = 10```
  - Start a process ```python3 cross-valid.py```
  - Validation results are saved into ```result/summary.csv```
  - Select a hyperparameter pair **(learning-rate, strength of regularization)** with the best accuracy
- Training/validation
  - Open ```cross-valid.py``` 
  - Set the parameters in dictionary ```pp``` as follows
    - ```saver = True```
    - ```is_optimal_h = True```
    - ```cnt_max_rand_search = 1```
    - Set ```labels``` to one of ```{'type', 'floor', 'position'}``` 
    - ```n_epoch = 50```
    - Set ```optimal_lr``` and ```optimal_reg``` to the optimal hyperparameters
    - Start a process ```python3 cross-valid.py```
- Test
  - Set ```labels``` to one of ```{'type', 'floor', 'position'}``` 
  - Start a process ```python3 test.py```
  - Test results are saved as ```result\test.csv```



### Results

- The results of all the tasks in the paper are uploaded [here](https://github.com/yodacatmeow/indoor-noise/tree/master/inter-floor-noise-classification/eusipco2020/results) (name of the folders represent the names of tasks in the paper)
- The files **cfm-sum-percent.csv** are the confusion matrices shown in the paper
- The files **h-rand-search-summary.csv** summarize the random search for finding the optimal hyperparameter pair



### Citing

When reporting results using this work, please cite:

```
H.Choi, H. Yang, S.Lee, and W. Seong (2020). Type/position classification of inter-floor noises in residential buildings with a single microphon via supervised learning. Manuscirpt submitted for 28th European Signal Processing Conference (EUSIPCO2020).
```

