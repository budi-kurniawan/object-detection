# Object Detection with A Pre-trained Model
This project shows how to use the Object Detection API and retrain RetinaNet to recognise Zombies using just the following 5 training images.

![Samples](samples.png?raw=true "samples")

These are the animated results:

![Results](zombie-anim.gif?raw=true "Animated gif")

## Pre-requisites and Running the Program
$ git clone --depth 1 https://github.com/tensorflow/models/ \
$ cd models/research/ \
$ protoc object_detection/protos/*.proto --python_out=. \
$ cd ../.. \
$ python -m pip install . \
$ wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz \
$ tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz \
$ mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint models/research/object_detection/test_data/ \
$ wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/zombie-walk-frames.zip \
    -O zombie-walk-frames.zip \
$ mkdir results \
$ unzip zombie-walk-frames.zip to ./results \
$ Run main.py \
