import os
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from common_functions import load_image_into_numpy_array, plot_detections, detect, build_detection_model

zombie_class_id = 1
# # define a dictionary describing the zombie class
category_index = {zombie_class_id: {'id': zombie_class_id, 'name': 'zombie'}}

tf.keras.backend.clear_session()
detection_model = build_detection_model()
checkpoint = tf.train.Checkpoint(detection_model)
checkpoint.restore('my-models/checkpoint/mycheckpoints-1')
print('checkpoint restored')

t2 = timer()
test_image_dir = './test-data/'
test_result_dir = './test-results/'

test_images_np = []

# load images into a numpy array
#for i in range(0, 237):
for i in range(0, 237):
    image_path = os.path.join(test_image_dir, 'zombie-walk' + "{0:04}".format(i) + '.jpg')
    test_images_np.append(np.expand_dims(load_image_into_numpy_array(image_path), axis=0))

label_id_offset = 1
results = {'boxes': [], 'scores': []}

print("start detection")
for i in range(len(test_images_np)):
    input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
    detections = detect(detection_model, input_tensor)
    plot_detections(
      test_images_np[i][0],
      detections['detection_boxes'][0].numpy(),
      detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
      detections['detection_scores'][0].numpy(),
      category_index, figsize=(15, 20), image_name=test_result_dir + "/gif_frame_" + ('%03d' % i) + ".jpg")
    results['boxes'].append(detections['detection_boxes'][0][0].numpy())
    results['scores'].append(detections['detection_scores'][0][0].numpy())

x = np.array(results['scores'])

# percent of frames where a zombie is detected
zombie_detected = (np.where(x > 0.9, 1, 0).sum())/237*100
print(f"zombie_detected: {zombie_detected}")
t3 = timer()
print("detection time (seconds):", timedelta(seconds=t3-t2))