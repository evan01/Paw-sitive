'''
    This is the starpoint for the cat/dog classifier program

'''
import tensorflow as tf
import os


'''
    PATHS
'''
TRAINING_IMAGES = './data/X_Train'
TESTING_IMAGES = './data/X_Test'
TRAINING_VALS = './data/Y_Train.csv'
TESTING_VALS = './data/Y_Test_example.csv'
LOGGING_PATH = './tensorFlowLog/'

'''
    TRAINING VARIABLES
'''
NUM_EPOCHS = 5 #We are going to run our dataset multiple times (epochs) through the neural net
SHUFFLE = True #Wen want to shuffle the images in our dataset multiple times

'''
    CODE
'''
def main():
    #First thing is to setup the tensor flow log session, so we can see how our network learns

    #Then get the list of filenames of the pictures to train data with
    filenames = [name for name in os.listdir(TRAINING_IMAGES)]

    #Pass the list of filenames and create a queue of successive tests to run
    queue = tf.train.string_input_producer(filenames,NUM_EPOCHS,SHUFFLE)

    #Create a reader, for the data we are interested in, ie pictures
    reader = tf.WholeFileReader

    #Then read the csv_file that has the outputs we're interested in
    name,file = tf.decode_csv(TRAINING_VALS)
    print("k")


main()