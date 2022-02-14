# Object Detection with A Pre-trained Model
This project shows how to use the Object Detection API and retrain RetinaNet to recognise Zombies using just the following 5 training images.

![Samples](samples.png?raw=true "samples")

These are the animated results:

![Results](zombie-anim.gif?raw=true "Animated gif")

The main.py script is a solution to a programming challenge from Coursera's "Advanced Computer Vision with TensorFlow" course, which provides the training and test samples for this project.

## Pre-requisites and Running the Program
Make sure tensorflow 2.7.0 and tensorflow-text 2.7.3 are used. TF 2.8 or different versions of tensorflow-text may cause errors.

### Extra Steps for Windows
$ On Windows, download https://github.com/google/protobuf/releases/download/v3.0.0-alpha-3/protoc-3.0.0-alpha-3-win32.zip and extract the zip file to install protoc.\
$ On Windows, Microsoft Visual C++ 14.0 or greater is required. It can be downloaded from https://visualstudio.microsoft.com/visual-cpp-build-tools/ \
$ On Windows, run the Visual C++ exe file, select Modify, select Development Tool and press the Modify button. \


### Setup
$ git clone this repository and cd to object-detection. \
$ git clone --depth 1 https://github.com/tensorflow/models/ \
$ cd models/research/ \
$ protoc object_detection/protos/*.proto --python_out=. \
$ cd ../.. \
$ cp setup.py models/research/ \
$ python -m pip install models/research \
$ wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz \
$ tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz \
$ mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/ \
$ wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/zombie-walk-frames.zip \
    -O zombie-walk-frames.zip \
$ mkdir results \
$ unzip zombie-walk-frames.zip to ./results \
$ Run main.py 
