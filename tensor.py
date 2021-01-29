
import tensorflow as tf

import sys
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import numpy as np
import cv2
from tensorflow_core.python.platform import flags
from tensorflow_core.python.training import monitored_session

from ocr_models import datasets, common_flags, data_provider
from ocr_models.datasets import quandm


FLAGS = flags.FLAGS
common_flags.define()

def load_images(image_buffer, batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    images_actual_data = np.ndarray(shape=(batch_size, height, width, 3), dtype='uint8')
    # pil_image = PIL.Image.open(tf.io.gfile.GFile(file_path, 'rb'))
    images_actual_data[0, ...] = np.asarray(image_buffer)
    return images_actual_data


def get_dataset_image_size(dataset_name):
  # Ideally this info should be exposed through the dataset interface itself.
  # But currently it is not available by other means.
  ds_module = getattr(datasets, dataset_name)
  height, width, _ = quandm.DEFAULT_CONFIG['image_shape']
  return width, height

path_to_save_pb = '/Users/tieungao/Codes/python/ai-research/yolov5/export_models/v2/saved_model.pb'
# with tf.Session(graph=tf.Graph()) as sess:
#     tf.saved_model.loader.load(
#         sess, [tf.saved_model.SERVING], path_to_save_pb)
#
#     ops = sess.graph.get_operations()
#     outputs_set = set(ops)
#     inputs = []
#     for op in ops:
#         if len(op.inputs) == 0 and op.type != 'Const':
#             inputs.append(op)
#         else:
#             for input_tensor in op.inputs:
#                 if input_tensor.op in outputs_set:
#                     outputs_set.remove(input_tensor.op)
#     outputs = list(outputs_set)
#     [x.name for x in outputs_set]




# graph_def = tf.GraphDef()
# with open(path_to_save_pb, 'rb') as f:
#     graph_def.ParseFromString(f.read())
#     for node in graph_def.node:
#         print(node.name)


# create frozen graph

meta_file = '/Users/tieungao/Codes/python/ai-research/yolov5/ocr52/model.ckpt-60404.meta'

checkpoint_dir = '/Users/tieungao/Codes/python/ai-research/yolov5/ocr52/model.ckpt-60404'
#
# with tf.Session(graph=tf.Graph()) as sess:
#     saver = tf.train.import_meta_graph(meta_file)
#     saver.restore(sess, checkpoint2)
#     output_graph_def = tf.graph_util.convert_variables_to_constants(
#         sess,
#         tf.get_default_graph().as_graph_def(),
#         [comma separated output nodes name]
#     )
#     # Saving "output_graph_def " in a file and generate frozen graph.
#     with tf.gfile.GFile('frozen_graph.pb', "wb") as f:
#         f.write(output_graph_def.SerializeToString())

# sess=tf.Session()
# saver = tf.train.import_meta_graph(meta_file)
# saver.restore(sess, checkpoint_dir)
#
# graph = tf.get_default_graph()
#
# input = graph.get_tensor_by_name('Placeholder:0')
# output = graph.get_tensor_by_name('AttentionOcr_v1/predicted_text:0')
#
# print(sess.run(output, feed_dict={input : images_data}))

def create_model(batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(
      num_char_classes=dataset.num_char_classes,
      seq_length=dataset.max_sequence_length,
      num_views=dataset.num_of_views,
      null_code=dataset.null_code,
      charset=dataset.charset)
  raw_images = tf.compat.v1.placeholder(
      tf.uint8, shape=[batch_size, height, width, 3])
  images = tf.map_fn(data_provider.preprocess_image, raw_images,
                     dtype=tf.float32)
  endpoints = model.create_base(images, labels_one_hot=None)
  return raw_images, endpoints


def predict(image_buffer):

    batch_size = 1
    dataset_name = 'quandm'
    checkpoint = '/Users/tieungao/Codes/python/ai-research/yolov5/ocr52/model.ckpt-60404'


    images_data = load_images(image_buffer, batch_size, dataset_name)


    images_placeholder, endpoints = create_model(batch_size, dataset_name)

    session_creator = monitored_session.ChiefSessionCreator(
      checkpoint_filename_with_path=checkpoint)
    with monitored_session.MonitoredSession(
          session_creator=session_creator) as sess:

        print(endpoints.predicted_text)
        predictions = sess.run(endpoints.predicted_text,
                               feed_dict={images_placeholder: images_data})

        print(images_placeholder)
        # print(images_data)

        op = sess.graph.get_operations()
        # for o in op:
        #     print(o.name)
        is_list = [pr_bytes.decode('utf-8') for pr_bytes in predictions.tolist()]

        pred = " ".join(is_list)

        print(pred)

    return pred

# image_buffer = cv2.imread('/Users/tieungao/Codes/python/ai-research/yolov5/test/01.png')
# predict(image_buffer)



