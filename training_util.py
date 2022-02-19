import csv
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf



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


#train_images_np, gt_boxes = get_training_data("./training/training-data.csv")

