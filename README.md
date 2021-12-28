# Object Detection with A Pre-trained Model
This project shows how to use the Object Detection API and retrain RetinaNet to recognise Zombies using just the following 5 training images.

![Samples](samples.png?raw=true "samples")

These are the animated results:

![Results](zombie-anim.gif?raw=true "Animated gif")

##Installation
$ git clone --depth 1 https://github.com/tensorflow/models/
$ cd models/research/ 
$ protoc object_detection/protos/*.proto --python_out=.
$ cd ../..
$ python -m pip install .

