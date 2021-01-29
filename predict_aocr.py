import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2

file_to_predict = '/content/drive/MyDrive/datasets/samples/orc2/crops/556.png'

path_to_pb = 'exported-model/frozen_graph.pb'

def getImage(path):
    with open(path, 'rb') as img_file:
        img = img_file.read()
    print(img)
    return img


with tf.Graph().as_default() as graph:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.gfile.GFile(path_to_pb, "rb") as f:

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                                name="", op_dict=None, producer_op_list=None)
            for op in graph.get_operations():
                print("Operation Name :" + op.name)
                print("Tensor Stats :" + str(op.values()))
            x = graph.get_tensor_by_name('input_image_as_bytes:0')
            y = graph.get_tensor_by_name('prediction:0')
            allProbs = graph.get_tensor_by_name('probability:0')

            img = getImage(file_to_predict)

            tf.global_variables_initializer()
            (y_out, probs_output) = sess.run([y, allProbs], feed_dict={
                x: [img]
            })
            print(y_out)
            print(probs_output)
            # print(allProbsToScore(probs_output))

            # return {
            #     "predictions": [{
            #         "ocr": str(y_out),
            #         "confidence": probs_output
            #     }]
            # };
