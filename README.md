# CIS4930: Deep Colorization with CNNs
This was a project made for a Deep Learning class. The project uses the [Georgia Tech Face Dataset](http://www.anefian.com/research/face_reco.htm). The CNN will take a greyscale image and then output a colorized version of it. The method for colorization and network architecture was based off of the SIGGRAPH paper ["Let there be Color!"](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/). 

___
## meanChrom.py
- Converts the image into L*A*B* format. The L* channel is inputting into the network, and then the network will output the overall mean values for the A* and B* channels.

## colorImg.py
- Extended version of `meanChrom.py`
- Takes a greyscale image and then outputs its best attempt at colorizing the image

## colorGPU.py
- Functions similarly to previous file, except it has been altered to run on CUDA
