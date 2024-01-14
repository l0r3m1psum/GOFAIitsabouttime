# [GOFAI](https://en.wikipedia.org/wiki/GOFAI) [itsabouttime](https://www.robots.ox.ac.uk/~vgg/research/time/)

This code aims to replicate the results of [It's About Time: Analog Clock
Reading in the Wild](https://www.robots.ox.ac.uk/~vgg/research/time/) using
classic computer vision algorithms.

The code expects that you have downloaded the data (following the instructions
on the paper's website) and put it in a directory called `itsabouttime/data`.

The code supports hot code reloading with `dlopen(3)` so no windows support
`:(`.

The code relies on multithreading for reading the CSV and classifing the images.
Make sure that you have compiled OpenCV with the right backend so that
`cv::getThreadNum()` gives a sensible answer.

# Resources

  * [LIST OF MAT TYPE IN OPENCV](https://gist.github.com/yangcha/38f2fa630e223a8546f9b48ebbb3e61a)
  * [List of OpenCv matrix types and mapping numbers](https://ros-developer.com/2017/12/04/list-opencv-matrix-types/)
  * [OpenCV's docs](https://docs.opencv.org/)
  * [Profiling OpenCV Applications](https://github.com/opencv/opencv/wiki/Profiling-OpenCV-Applications)
