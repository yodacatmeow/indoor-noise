# CS-APT

**CS-APT** is an inter-floor noise dataset collected in CHULSAN-JOOGONG APT. A single microphone in a smartphone (Samgsung Galaxy S6) was used as a receiver to record inter-floor noises. Inter-floor noises were sampled at 44,100 Hz for ~5 s.

**CS-APT** includes 4 different inter-floor noise types. They are: medicine ball falling on the floor from a height of 1.2 m (**MB**), a hammer dropped from 1.2 m above the floor (**HD**), hammering (**HH**), and dragging a chair (**CD**).

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/CS-APT/figure/noise_type_v0.png)

Inter-floor noises were generated on the upper floor, lower floor, and above two floors based on the position of the receiver.

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/CS-APT/figure/cs-apt-size_v4.png)

![](https://github.com/yodacatmeow/indoor-noise/blob/master/indoor-noise-set/CS-APT/figure/table_drawing.png)

(Recv 0: Galaxy-S6-UWAL, Recv 1: Galaxy-S6-SLEE)

An machine learning method was evaluated on this dataset:

| Title                                                        | Paper                                   |                             Code                             |
| ------------------------------------------------------------ | --------------------------------------- | :----------------------------------------------------------: |
| Type/position classification of inter-floor noises in residential buildings with a single microphone via supervised learning | choi2020a (*Submitted for EUSIPCO2020*) | [GitHub Repo.](https://github.com/yodacatmeow/indoor-noise/tree/master/inter-floor-noise-classification/eusipco2020) |

Metadata of this dataset is available [here](https://github.com/yodacatmeow/indoor-noise/blob/master/inter-floor-noise-classification/eusipco2020/metadata.csv) and each column in the metadata represents:

- track-id: [digital-id in the table in the above]-[sample number]
- building: 'C'=CS-APT
- type: inter-floor noise type
- position: [source]-[floor]-[XY position]-[receiver]-[floor]-[XY position]

### Citing

When reporting results using this dataset, please cite:

```
H. Choi, H. Yang, S. Lee, and W. Seong (2020). Type/position classification of inter-floor noises in residential buildings with a single microphon via supervised learning. Manuscirpt submitted for 28th European Signal Processing Conference (EUSIPCO2020).
```

