#
import cv2
import os
import tensorflow as tf
import csv
from shutil import copyfile
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)

def main():
    '''
    Wanted to separate the cats from the dogs, for training of the inception network
    :return: 
    '''

    #First thing is to get all the dog and cat pictures
    catFiles = []
    dogFiles = []
    with open('./data/Y_Train.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[1]=='1':
                dogFiles.append(row[0])
            else:
                catFiles.append(row[0])

    #Then read the actual files and place them in the cats or dogs directory
    catFiles = catFiles[1:]
    dogFiles = dogFiles[1:]
    for i in catFiles:
        copyfile('./data/X_Train/'+i,"./data/Cats/"+i)
    for j in dogFiles:
        copyfile('./data/X_Train/'+i,"./data/Dogs/"+j)
    print("k")

def testTensors():
    n1 = tf.constant(3.0,tf.float32)
    n2 = tf.constant(4.0)
    sess = tf.Session()
    n3 = tf.subtract(n1,n2)
    print ("node3: ",n3)
    print("sess run 3",sess.run(n3))

    # Create summary data
    file_write = tf.summary.FileWriter('./tensorFlowLog/',sess.graph)
    print(sess.run([n1,n2]))
    print(n1,n2)

main()
