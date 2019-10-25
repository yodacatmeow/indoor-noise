# Indoor-noise

In residential buildings, noise generated by residents or home appliances propagates through building structure and annoys residents on other floors. It is difficult for human to precisely identify the type/position of inter-floor noise, and some conflicts between residents have originated from this incorrect estimation of type/position. Correctly **identifying the noise type and position** is considered to be the first step in solving noise problem.

We built [three different datasets](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set) to study this problem.

| Name                                                         | Building type      |
| ------------------------------------------------------------ | ------------------ |
| [SNU-B36-50E](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/SNU-B36-50E) | Office building    |
| [CS-APT](https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/CS-APT) | Apartment building |
| [BDML-APT]( https://github.com/yodacatmeow/indoor-noise/tree/master/indoor-noise-set/BDML-APT) | Apartment building |



### Single sensor acoustical approach

A noise signal over a single microphone with a sufficient duration might contain the dispersive nature of the plate wave or unidentified features.



### Study

- [IWAENC 2018](https://ieeexplore.ieee.org/abstract/document/8521392) ([repo.](https://github.com/yodacatmeow/VGG16-SNU-B36-50))
  - In this work, an method for inter-floor noise type/position classification was proposed and validated against an inter-floor noise dataset.
  - We built an inter-floor noise dataset SNU-B36-50 in an office building.

- [Appl. Sci.](https://www.mdpi.com/2076-3417/9/18/3735) (repo.)
  - This paper is expanded version of IWAENC 2018.
  - We collected inter-floor noise(SNU-B36-50E) at many positions around the learned positions.
  - Type/position classification of noise at unlearned position was shown.

- ICASSP 2020 (*submitted*, repo.)
  - In this paper, the generalizability of the proposed method in Appl. Sci. was addressed against inter-floor noise in residential buildings.
  - We collected inter-floor noise in two apartment buildings.

- Mobile application (*iOS*, repo.)



(To be updated ..)



## References