import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf
from common_functions import build_detection_model, predict_and_plot

tf.get_logger().setLevel('ERROR')
tf.keras.backend.clear_session()

experiment_name = 'octopus1'
zombie_class_id = 1
# # define a dictionary describing the zombie class
category_index = {zombie_class_id: {'id': zombie_class_id, 'name': 'zombie'}}
detection_model = build_detection_model()
checkpoint = tf.train.Checkpoint(detection_model)
checkpoint.restore('experiments/' + experiment_name + '/my-models/checkpoint/mycheckpoints-1')
print('checkpoint restored')

t1 = timer()
test_image_dir = './experiments/' + experiment_name + '/test-data/'
test_result_dir = './experiments/' + experiment_name + '/test-results/'
results, num_tests = predict_and_plot(detection_model, test_image_dir, test_result_dir, category_index)
print('num tests:', num_tests)
x = np.array(results['scores'])
# percent of frames where a zombie is detected
zombie_detected = (np.where(x > 0.9, 1, 0).sum()) / num_tests * 100
print(f"zombie_detected: {zombie_detected}")
t2 = timer()
print("detection time (seconds):", timedelta(seconds=t2-t1))