# SNU-B36-50E

- SNU-B36-50E is an inter-floor noise dataset gathered in Building 36 at Seoul National University.
- SNU-B36-50 (*older version with smaller number of audio clips*) is merged and rearranged into here.
- Inter-floor noises are recored with a single microphone of a smartphone (Samsung Galaxy S6).
- The sampling frequency is set as 44,100 Hz.



## Category

- Each audio clip in the dataset has two labels: noise type and noise source position
  - Noise type (5 types)
    - MB: a medicine ball at height 1.2 m above the floor falls and hits the floor
    - HD: a hammer at height 1.2 m above the floor falls and hits the floor
    - HH: Hammering
    - CD: chair dragging
    - VC: vacuum cleaner
  - Noise source position (19 positions)
    - 1F0m, 1F6m, 1F12m, 2F0m, 2F6m, 2F12m, 3F0m, 3F1m, 3F2m, 3F3m, 3F4m, 3F5m, 3F6m, 3F7m, 3F8m , 3F9m, 3F10m, 3F11m, 3F12m



## Citing

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


## TODO

- Metadata: event_start_s, event_end_s