from ctypes import util
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from training_util import get_training_data, load_image_into_numpy_array, plot_detections

# import glob
# import imageio
# from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from object_detection.utils import config_util # module for reading and updating configuration files.
from object_detection.builders import model_builder

train_images_np, gt_boxes = get_training_data("./training/training-data.csv")
zombie_class_id = 1 # define the category index dictionary
category_index = {zombie_class_id: {'id': zombie_class_id, 'name': 'zombie'}} # define a dictionary describing the zombie class
num_classes = 1 # Specify the number of classes that the model will predict

# Data preprocessing
# The `label_id_offset` here shifts all classes by a certain number of indices;
# we do this here so that the model receives one-hot labels where non-background
# classes start counting at the zeroth index.  This is ordinarily just handled
# automatically in our training binaries, but we need to reproduce it here.
label_id_offset = 1
train_image_tensors = []

# lists containing the one-hot encoded classes and ground truth boxes
gt_classes_one_hot_tensors = []
gt_box_tensors = []

for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):  
    # convert training image to tensor, add batch dimension, and add to list
    train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
        train_image_np, dtype=tf.float32), axis=0))
    
    # convert numpy array to tensor, then add to list
    gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
    
    # apply offset to to have zero-indexed ground truth classes
    zero_indexed_groundtruth_classes = tf.convert_to_tensor(
        np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
    
    # do one-hot encoding to ground truth classes
    gt_classes_one_hot_tensors.append(tf.one_hot(
        zero_indexed_groundtruth_classes, num_classes))

print('Done prepping data.')

# Visualise zombies with ground truth bounding boxes
path_to_bounding_boxes = "./saved_images/bounding_boxes.png"
my_file = Path(path_to_bounding_boxes)
if not my_file.is_file():
    dummy_scores = np.array([1.0], dtype=np.float32)
    # define the figure size
    plt.figure(figsize=(30, 15))
    # use the `plot_detections()` utility function to draw the ground truth boxes
    for idx in range(len(train_images_np)):
        plt.subplot(2, 4, idx+1)
        plot_detections(train_images_np[idx], gt_boxes[idx],
            np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32), dummy_scores, category_index)
    plt.savefig(path_to_bounding_boxes)

# Download the checkpoint containing the pre-trained weights
# Configure the models
tf.keras.backend.clear_session()

# define the path to the .config file for ssd resnet 50 v1 640x640
pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

# Load the configuration file into a dictionary
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
# Modify the number of classes from its default of 90
model_config.ssd.num_classes = num_classes

# Freeze batch normalization
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(model_config=model_config, is_training=True) # build the model

tmp_box_predictor_checkpoint = tf.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
)

tmp_model_checkpoint = tf.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=tmp_box_predictor_checkpoint)

# restore checkpoints
checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'
# Define a checkpoint that sets `model= None
checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)

# Restore the checkpoint to the checkpoint path
#checkpoint.restore(checkpoint_path)
checkpoint.restore(checkpoint_path).expect_partial()

tmp_image, tmp_shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))

# run a prediction with the preprocessed image and shapes
tmp_prediction_dict = detection_model.predict(tmp_image, tmp_shapes)

# postprocess the predictions into final detections
tmp_detections = detection_model.postprocess(tmp_prediction_dict, tmp_shapes)

print('Weights restored!')

# Eager mode custom training loop
batch_size = 4
num_batches = 100
learning_rate = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

# define a list that contains the layers that you wish to fine tune
to_fine_tune = []
for v in detection_model.trainable_variables:
  if v.name.startswith('WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutional'):
    to_fine_tune.append(v)

# define the training step
@tf.function
def train_step_fn(image_list, groundtruth_boxes_list, groundtruth_classes_list, model,
                optimizer, vars_to_fine_tune):
    """A single training iteration.
    Args:
      image_list: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    with tf.GradientTape() as tape:
    ### START CODE HERE (Replace instances of `None` with your code) ###
        # Preprocess the images
        preprocessed_image_list = []
        true_shape_list = []
        for image in image_list:
          tmp_image, tmp_shape = model.preprocess(image)
          preprocessed_image_list.append(tmp_image)
          true_shape_list.append(tmp_shape)

        preprocessed_image_tensor = tf.concat(preprocessed_image_list, axis=0)
        true_shape_tensor = tf.concat(true_shape_list, axis=0)

        prediction_dict = model.predict(preprocessed_image_tensor, true_shape_tensor)
        model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list, groundtruth_classes_list=groundtruth_classes_list)          

        try:
          losses_dict = model.loss(prediction_dict, true_shape_tensor)
        except Exception as e:
          print('--------- error calculating losses_dict ------', e)
        # Calculate the total loss (sum of both losses)
        print('losses_dict:', losses_dict)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        # Calculate the gradients
        gradients = tape.gradient(total_loss, vars_to_fine_tune)
        # Optimize the model's selected variables
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))        
    return total_loss

print('Start fine-tuning!', flush=True)

tf.saved_model.save(detection_model, 'my-models/model0')
print('model0 saved')
t1 = timer()

for idx in range(num_batches):
    # Grab keys for a random subset of examples
    all_keys = list(range(len(train_images_np)))
    random.shuffle(all_keys)
    example_keys = all_keys[:batch_size] # budi: take 4 of 5 indexes

    # Get the ground truth
    gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
    gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
    
    # get the images
    image_tensors = [train_image_tensors[key] for key in example_keys]

    # Training step (forward pass + backwards pass)
    total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list,
        detection_model, optimizer, to_fine_tune)

    if idx % 10 == 0:
      print('batch ' + str(idx) + ' of ' + str(num_batches) + ', loss=' +  str(total_loss.numpy()), flush=True)

print('Done fine-tuning!')
tf.saved_model.save(detection_model, 'my-models/model1')
print('model1 saved')
t2 = timer()

test_image_dir = './results/'
test_images_np = []

# load images into a numpy array. this will take a few minutes to complete.
for i in range(0, 237):
    image_path = os.path.join(test_image_dir, 'zombie-walk' + "{0:04}".format(i) + '.jpg')
    print(image_path)
    test_images_np.append(np.expand_dims(
      load_image_into_numpy_array(image_path), axis=0))

# Preprocess, predict, postprocess images
# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor):
    """Run detection on an input image.
    Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

    Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)    
    # use the detection model's postprocess() method to get the the final detections
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

label_id_offset = 1
results = {'boxes': [], 'scores': []}

for i in range(len(test_images_np)):
    input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
    detections = detect(input_tensor)
    plot_detections(
      test_images_np[i][0],
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
      detections['detection_scores'][0].numpy(),
      category_index, figsize=(15, 20), image_name="./results/gif_frame_" + ('%03d' % i) + ".jpg")
    results['boxes'].append(detections['detection_boxes'][0][0].numpy())
    results['scores'].append(detections['detection_scores'][0][0].numpy())

x = np.array(results['scores'])

# percent of frames where a zombie is detected
zombie_detected = (np.where(x > 0.9, 1, 0).sum())/237*100
print(f"zombie_detected: {zombie_detected}")
t3 = timer()
print("training time (seconds):", timedelta(seconds=t2-t1))
print("detection time (seconds):", timedelta(seconds=t3-t2))