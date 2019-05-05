# SNU-B36-50E

**SNU-B36-50E** is an inter-floor noise dataset collected in building 36 at Seoul National University. **SNU-B36-50** (*older version with smaller number of audio clips*) is rearranged into here. A single microphone in a smartphone (Samgsung Galaxy S6) was used as a receiver to record inter-floor noises. Inter-floor noises were sampled at 44,100 Hz for ~5 s.

The inter-floor noises included in this dataset can be classified into 5 types or 19 positions. The inter-floor noise types are a medicine ball falling to the floor from a height of 1.2 m (**MB**), a hammer falling from 1.2 m above the floor (**HD**),  hammering (**HH**), dragging a chair (**CD**), and a running vacuum cleaner (**VC**).

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/SNU-B36-50E/figure/noise_type.png)

*(ToDo: img_19 positions)*

Each audio clip can be labeled as a noise type and a position, where the first row of the following table denotes the distance from the origin along the X axis. And the digits in the table (e.g. 001) denote names of folders in [SNU-B36-50E/audio](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/SNU-B36-50E/audio). Each folder in this path contains 50 inter-floor noises.

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/SNU-B36-50E/figure/categories.jpeg)

## Citing

When reporting results using this dataset, please cite:

```
@inproceedings{choi2018floornoise,
  title={Classification of noise between floors in a building using pre-trained deep convolutional neural networks},
  author={Choi, Hwiyong and Lee, Seungjun and Yang, Haesang and Seong, Woojae},
  booktitle={2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC)},
  pages={535--539},
  year={2018},
  organization={IEEE}
}
```



## ToDo

- Metadata: event_start_s, event_end_s
- Citing