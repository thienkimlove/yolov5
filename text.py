import cv2

import numpy as np

img = 'temp/8.jpg'

model_path = '/Users/tieungao/Codes/python/ai-research/openvino-latest/models/downloaded/intel/text-detection-0004/FP32/text-detection-0004'


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    # text_classes = output['model/segm_logits/add']
    text_classes = output[0]
    # TODO 2: Resize this output back to the size of the input

    print("Text_classes")
    print(text_classes.shape)

    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])
    return out_text


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "TEXT":
        return handle_text
    else:
        return None


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))
    return mask


def create_output_image(model_type, image, output):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    if model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1]>0.5, 100, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
        return image
    else:
        print("Unknown model type, unable to create output image.")
        return image


# Load the model
net = cv2.dnn.readNet(model_path + '.xml', model_path + '.bin')

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


frame = cv2.imread(img)

blob = cv2.dnn.blobFromImage(frame, size=(1280, 768), ddepth=cv2.CV_8U)
net.setInput(blob)
out = net.forward()

processed_func = handle_output("TEXT")
processed_output = processed_func(out, frame.shape)
# Create an output image based on network
try:
    output_image = create_output_image("TEXT", frame, processed_output)
    print("Success!")
except:
    output_image = frame
    print("Error!")
# Save down the resulting image
cv2.imwrite("{}-output.png".format("TEXT"), output_image)