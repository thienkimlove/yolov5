import os
import cv2
import random
import numpy as np 
import pandas as pd
import tensorflow as tf

ANNOTATION_FILE = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/anov2_file.csv'
CROP_DIR = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/crops'
IMAGE_DIR = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/images'

MAX_STR_LEN = 20
null = 43


def get_char_mapping():
    label_file = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/charset-label.txt'
    with open(label_file, "r") as f:
        char_mapping = {}
        rev_char_mapping = {}
        for line in f.readlines():
            m, c = line.split("\n")[0].split("\t")
            char_mapping[c] = m
            rev_char_mapping[m] = c
    return char_mapping, rev_char_mapping


def read_image(img_path):
    return cv2.imread(img_path)


def padding_char_ids(char_ids_unpadded, null_id = null, max_str_len=MAX_STR_LEN):
    return char_ids_unpadded + [null_id for x in range(max_str_len - len(char_ids_unpadded))]


def get_bytelist_feature(x):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=x))


def get_floatlist_feature(x):
    return tf.train.Feature(float_list = tf.train.FloatList(value=x))


def get_intlist_feature(x):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[int(y) for y in x]))


def _float_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """



  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def get_tf_example(img_file, annotation, num_of_views=1):
	# img_array = read_image(img_file)
	# img = gfile.FastGFile(img_file, 'rb').read()
	char_map, _ = get_char_mapping()

	split_text = [x for x in annotation]
	char_ids_unpadded = [char_map[x] for x in split_text]
	char_ids_padded = padding_char_ids(char_ids_unpadded)
	char_ids_unpadded = [int(x) for x in char_ids_unpadded]
	char_ids_padded = [int(x) for x in char_ids_padded]

	# Create a generic TensorFlow-based utility for converting all image codings.
	coder = ImageCoder()

	try:
		image_buffer, height, width = _process_image(img_file, coder)
	except Exception as e:
		print(e)
		return None

	#
	#
	# features = tf.train.Features(feature = {
	# 	'image/format': get_bytelist_feature([b'png']),
	# 	'image/encoded': get_bytelist_feature([img]),
	# 	'image/class': get_intlist_feature(char_ids_padded),
	# 	'image/unpadded_class': get_intlist_feature(char_ids_unpadded),
	# 	# 'image/height': get_intlist_feature([img_array.shape[0]]),
	# 	'image/width': get_intlist_feature([img_array.shape[1]]),
	# 	'image/orig_width': get_intlist_feature([img_array.shape[1]/num_of_views]),
	# 	'image/text': get_bytelist_feature([annotation.encode('utf-8')])
	# 	}
	# )
	#
	# example = tf.train.Example(features=features)

	colorspace = 'RGB'
	channels = 3
	image_format = 'JPEG'

	# example = tf.train.Example(features=tf.train.Features(feature={
	# 	'image/height': _int64_feature(height),
	# 	'image/width': _int64_feature(width),
	# 	'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
	# 	'image/channels': _int64_feature(channels),
	# 	'image/class/label': _int64_feature(label),
	# 	'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
	# 	'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
	# 	'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
	# 	'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
	# return example

	example = tf.train.Example(features=tf.train.Features(
		feature={
			'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
			'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
			'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
			'image/channels': _int64_feature(channels),
			'image/class': _int64_feature(char_ids_padded),
			'image/unpadded_class': _int64_feature(char_ids_unpadded),
			'image/height': _int64_feature(height),
			'image/width': _int64_feature(width),
			'image/orig_width': get_intlist_feature([width / num_of_views]),
			'image/text': _bytes_feature(tf.compat.as_bytes(annotation))
		}
	))

	return example


def get_tf_records(train_total, test_total):
	train_file = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/train.tfrecord'
	test_file = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/test.tfrecord'
	valid_file = '/Users/tieungao/Codes/python/ai-research/yolov5/datav2/valid.tfrecord'
	if os.path.exists(train_file):
		os.remove(train_file)
	if os.path.exists(test_file):
		os.remove(test_file)
	if os.path.exists(valid_file):
		os.remove(valid_file)
	train_writer = tf.io.TFRecordWriter(train_file)
	test_writer = tf.io.TFRecordWriter(test_file)
	valid_writer = tf.io.TFRecordWriter(valid_file)
	annot = pd.read_csv(ANNOTATION_FILE)
	files = list(annot['files'].values)
	random.shuffle(files)

	record_train = 0
	record_test = 0
	record_valid = 0

	for i, file in enumerate(files):
		print('writing file:', file)
		annotation = annot[annot['files'] == file]
		annotation = annotation['text'].values[0]
		example = get_tf_example(IMAGE_DIR + '/' + file, annotation)
		if i < train_total:
			train_writer.write(example.SerializeToString())
			record_train += 1
		elif i < test_total:
			test_writer.write(example.SerializeToString())
			record_test += 1
		else:
			valid_writer.write(example.SerializeToString())
			record_valid += 1

	train_writer.close()
	test_writer.close()
	valid_writer.close()

	print("TOtal train {}".format(record_train))
	print("TOtal test {}".format(record_test))
	print("TOtal valid {}".format(record_valid))


if __name__ == '__main__':
	get_tf_records(400, 470)





