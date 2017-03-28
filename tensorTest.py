#
import cv2
import tensorflow as tf

def main():
    print("k")
    testTensors()

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
