# BDML-APT

**BDML-APT** is an inter-floor noise dataset collected in [BADA-MAEUL APT]([https://www.google.com/maps/place/%EB%B0%94%EB%8B%A4%EB%A7%88%EC%9D%84%EC%95%84%ED%8C%8C%ED%8A%B8/@37.5025693,126.9154497,15z/data=!4m5!3m4!1s0x0:0x75d90e5df5835799!8m2!3d37.5025693!4d126.9154497](https://www.google.com/maps/place/바다마을아파트/@37.5025693,126.9154497,15z/data=!4m5!3m4!1s0x0:0x75d90e5df5835799!8m2!3d37.5025693!4d126.9154497)). A single microphone in a smartphone (Samsung Galaxy S6) was used as a receiver to record inter-floor noises. Inter-floor noises were sampled at 44,100 Hz for ~5 s.

**BDML-APT** includes 5 different inter-floor noise types. They are: medicine ball falling on the floor from a height of 1.2 m (**MB**), a hammer dropped from 1.2 m above the floor (**HD**), hammering (**HH**), dragging a chair (**CD**), and running a vacuum cleaner (**VC**).

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/SNU-B36-50E/figure/noise_type.png)

Inter-floor noises were generated on the upper floor, lower floor, and above two floors based on the position of the receiver.

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/BDML-APT/figure/bdml-apt-size_v4.png)

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/BDML-APT/figure/table_drawing.png)

| Title                                                        | Paper                                   |                             Code                             |
| ------------------------------------------------------------ | --------------------------------------- | :----------------------------------------------------------: |
| Type/position classification of inter-floor noises in residential buildings with a single microphone via supervised learning | choi2020a (*Submitted for EUSIPCO2020*) | [GitHub Repo.]( https://github.com/yodacatmeow/indoor-noise/tree/master/inter-floor-noise-classification/eusipco2020) |

Metadata of this dataset is available [here](https://github.com/yodacatmeow/indoor-noise/blob/master/inter-floor-noise-classification/eusipco2020/metadata.csv) and each column in the metadata represents:

- track-id: [digital-id in the table in the above]-[sample number]
- building: 'B'=BDML-APT
- type: inter-floor noise type
- position: [source]-[floor]-[XY position]-[receiver]-[floor]-[XY position]



### Citing

When reporting results using this dataset, please cite:

```
H. Choi, H. Yang, S. Lee, and W. Seong (2020). Type/position classification of inter-floor noises in residential buildings with a single microphon via supervised learning. Manuscirpt submitted for 28th European Signal Processing Conference (EUSIPCO2020).
```

