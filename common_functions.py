import csv
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils # module for visualization
from object_detection.utils import config_util
from object_detection.builders import model_builder


# configure plot settings via rcParams
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = False
plt.rcParams['ytick.labelsize'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['figure.figsize'] = [14, 7]

def plot_detections(image_np, boxes, classes, scores, category_index, figsize=(12, 16), image_name=None):
    """Wrapper function to visualize detections.
    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
          and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
          this function assumes that the boxes to be plotted are groundtruth
          boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """    
    image_np_with_annotations = image_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations, boxes, classes, scores, category_index,
        use_normalized_coordinates=True, min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def get_training_data(training_info_file):
    train_images_np = []
    gt_boxes = []
    with open(training_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            image_path = row[0]
            y1 = float(row[1])
            x1 = float(row[2])
            y2 = float(row[3])
            x2 = float(row[4])
            train_images_np.append(load_image_into_numpy_array(image_path))
            gt_boxes.append(np.array([[y1, x1, y2, x2]]))
    return train_images_np, gt_boxes

# Uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(model, input_tensor):
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
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections


def build_detection_model():
    # the number of classes that the model will predict
    num_classes = 1
    # define the path to the .config file for ssd resnet 50 v1 640x640
    pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    # Modify the number of classes from its default of 90
    model_config.ssd.num_classes = num_classes

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True
    # build the model
    detection_model = model_builder.build(model_config=model_config, is_training=True)

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
    return detection_model

