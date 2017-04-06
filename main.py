import numpy as np
import csv
from tqdm import tqdm
import multiprocessing
import os
import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

'''
    PATHS
'''
TESTING_IMAGES = './data/X_Test'
TESTING_VALS = './data/Y_Test_example.csv'
LOGGING_PATH = './tensorFlowLog'
MODEL_PATH = './data/CNN_Models/output_graph.pb'
RESULTS_PATH = './data/testResults/output_labels.csv'


def displayImage(img, desc):
    img = cv2.resize(img, (int(img.shape[0] / 4), int(img.shape[1] / 4)))
    img = cv2.putText(img, str(desc), (619, 351), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10, cv2.LINE_AA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)


def create_CNNetwork():
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def guessImage(imPath):
    """
    When you want to guess just one image
    """
    answer = None
    # start up tensor flow
    create_CNNetwork()
    string = ""
    if not tf.gfile.Exists(imPath):
        tf.logging.fatal("File doesn't exist")
        tf.logging.fatal(imPath)
        exit(1)

    with tf.Session() as sess:
        # aquire the tensor of the CNN we want to check
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        image_data = tf.gfile.FastGFile(imPath, 'rb').read()
        predictions = sess.run(
            softmax_tensor,
            {'DecodeJpeg/contents:0': image_data}
        )
        guess = np.squeeze(predictions)
        if (guess[0] > guess[1]):
            # Then the net thinks it's a cat
            answer = "Cat"
        else:
            # Then the net thinks it's a dog
            answer = "Dog"
        print('This is a picture of a: ' + answer)
        print("Confidence Cat: " + str(guess[0]))
        print("Confidence Dog: " + str(guess[1]))
        print(" ")
        string = answer

    im = cv2.imread(imPath)
    displayImage(im, str(string))

% pylab
% matplotlib inline
plt.rcParams['figure.figsize'] = (20, 20.0)
path1 = "./data/myImagesOfGatsby/IMG_0037-2.jpg"

# guessImage(path1)
guessImage(path1)