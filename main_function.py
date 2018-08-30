import numpy as np
import os
import scipy.misc as misc
import random
import tensorflow as tf
import numpy as np
import BuildNetVgg16
import CheckVGG16Model
import scipy.misc as misc
import pandas as pd
import Data_Reader
import cv2
import time 
import glob
import matplotlib.pyplot as plt
from skimage.draw import polygon
import scipy.misc as misc
import random
import os


def generate_label_BI():
    path = "../BrandImages/"
    for i in range(1,11):
        directory = path+str(i)+"/*.jpg"
        file_list = glob.glob(directory)
        print(str(i)+"...")
        for j in file_list:
            name = j.split("/")[-1][0:-4]
            img = cv2.imread(j)
            label = np.ones((img.shape[0],img.shape[1],1), dtype=int)
            if(i!=1):
                label = np.multiply(label,i)
            cv2.imwrite("../data/label/"+name+".png",label)
            cv2.imwrite("../data/input/"+name+".jpg",img)
            
def ret_shape(path):
    img = cv2.imread(path)
    return img.shape
def brand_annotation():
    b_annotation = dict()
    path = "../BrandImagesFromShelves/"
    path_2 = "../ShelfImages/"
    for i in range(1,11):
        print(str(i)+"..")
        fld_path = path+str(i)+"/"
        list_val = os.listdir(fld_path)
        for j in list_val:
            name = j.split(".")[0]+".JPG"
            a = ret_shape(fld_path+j)
            x = j.split(".")[1].split("_")
            x[0] = str(i)
            x[4] = str(a[0])
            #x = np.asarray(x, dtype=int)
            if name in b_annotation.keys():  
                b_annotation[name].append(x)  
            else:  
                b_annotation[name] =  [x]  
    return b_annotation

def generate_label_SI(annotation):
    path = "../ShelfImages/"
    count = 0
    empty_list = []
    file_list = [each for each in os.listdir(path) if each.endswith('.JPG')]
    for j in file_list:
        count+=1
        if(count%100 == 0):
            print(count)
        name = j
        #print(name)
        img = cv2.imread(path+j)
        label = np.zeros((img.shape[0],img.shape[1]), dtype=int)
        if name in annotation.keys():
            box = annotation[name]
            for i in box:
                i = np.asarray(i,dtype = int)
                c = np.array([i[1], i[1]+i[3],i[1]+i[3],i[1]])
                r = np.array([i[2], i[2],i[2]+i[4],i[2]+i[4]])
                rr, cc = polygon(r, c)
                label[rr, cc] = i[0]
        else:
            empty_list.append(name)
            
        cv2.imwrite("../data/test_label/"+name[0:-4]+".png",label)
        #cv2.imwrite("./data/input/"+name+".jpg",img)
    return empty_list

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

print("Creting input data and corresponding label")
data_dir = "../data/"
if not os.path.exists(data_dir+"input/"): os.makedirs(data_dir+"input/")
if not os.path.exists(data_dir+"label/"): os.makedirs(data_dir+"label/") 
generate_label_BI()

print("Creting test data and corresponding label")
annotation = brand_annotation()

data_dir = "../data/"

if not os.path.exists(data_dir+"test_label/"): os.makedirs(data_dir+"test_label/") 
emp_list = generate_label_SI(annotation)

#...........................................Input and output folders.................................................
Train_Image_Dir="../data/input" 
# Images and labels for training
# lables have to be created 
Train_Label_Dir="../data/label"
# Annotetion in png format for train images and validation images (assume the name of the images and annotation images are the same (but annotation is always png format))
UseValidationSet = True
# do you want to use validation set in training
Valid_Image_Dir="../ShelfImages"
# Validation images that will be used to evaluate training
Valid_Labels_Dir="../data/test_label"
#  (the  Labels are in same folder as the training set)
logs_dir= "./logs/"
# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
model_path="../vgg16.npy"
# "Path to pretrained vgg16 model for encoder"
learning_rate=1e-5 
#Learning rate for Adam Optimizer
CheckVGG16Model.CheckVGG16(model_path)


TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Batch_Size=1 # Number of files per training iteration
Weight_Loss_Rate=5e-4# Weight for the weight decay loss function
MAX_ITERATION = int(90000) # Max  number of training iteration
NUM_CLASSES = 11

def main(argv=None):
    tf.reset_default_graph()
    keep_prob= tf.placeholder(tf.float32, name="keep_probabilty") #Dropout probability
#.........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") #Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    GTLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="GTLabel")#Ground truth labels for training
  #.........................Build FCN Net...............................................................................................
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) #Create class for the network
    Net.build(image, NUM_CLASSES,keep_prob)# Create the net and load intial weights
#......................................Get loss functions for neural net work  one loss function for each set of label....................................................................................................
    Loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(GTLabel, squeeze_dims=[3]), logits=Net.Prob,name="Loss")))  # Define loss function for training
   #....................................Create solver for the net............................................................................................
    trainable_var = tf.trainable_variables() # Collect all trainable variables for the net
    train_op = train(Loss, trainable_var) #Create Train Operation for the net
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    TrainReader = Data_Reader(Train_Image_Dir,  GTLabelDir=Train_Label_Dir,BatchSize=Batch_Size) #Reader for training data
    if UseValidationSet:
        ValidReader = Data_Reader(Valid_Image_Dir,  GTLabelDir=Valid_Labels_Dir,BatchSize=Batch_Size) # Reader for validation data
    sess = tf.Session() #Start Tensorflow session
# -------------load trained model if exist-----------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer()) #Initialize variables
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------

    f = open(TrainLossTxtFile, "w")
    f.write("Iteration\tloss\t Learning Rate="+str(learning_rate))
    f.close()
    if UseValidationSet:
        f = open(ValidLossTxtFile, "w")
        f.write("Iteration\tloss\t Learning Rate=" + str(learning_rate))
        f.close()
#..............Start Training loop: Main Training....................................................................
    for itr in range(MAX_ITERATION):
        Images,  GTLabels =TrainReader.ReadNextBatchClean() # Load  augmeted images and ground true labels for training
        feed_dict = {image: Images,GTLabel:GTLabels, keep_prob: 0.5}
        sess.run(train_op, feed_dict=feed_dict) # Train one cycle
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 10000 == 0 and itr>0:
            print("Saving Model to file in "+logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr) #Save model
#......................Write and display train loss..........................................................................
        if itr % 10==0:
            # Calculate train loss
            feed_dict = {image: Images, GTLabel: GTLabels, keep_prob: 1}
            TLoss=sess.run(Loss, feed_dict=feed_dict)
            print("Step "+str(itr)+" Train Loss="+str(TLoss))
            #Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write("\n"+str(itr)+"\t"+str(TLoss))
                f.close()
#......................Write and display Validation Set Loss by running loss on all validation images.....................................................................
        if UseValidationSet and itr % 2000 == 0 and itr>1:
            SumLoss=np.float64(0.0)
            NBatches=np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
            print("Calculating Validation on " + str(ValidReader.NumFiles) + " Images")
            for i in range(NBatches):# Go over all validation image
                Images, GTLabels= ValidReader.ReadNextBatchClean() # load validation image and ground true labels
                feed_dict = {image: Images,GTLabel: GTLabels ,keep_prob: 1.0}
                # Calculate loss for all labels set
                TLoss = sess.run(Loss, feed_dict=feed_dict)
                SumLoss+=TLoss
                NBatches+=1
            SumLoss/=NBatches
            print("Validation Loss: "+str(SumLoss))
            with open(ValidLossTxtFile, "a") as f:
                f.write("\n" + str(itr) + "\t" + str(SumLoss))
                f.close()

main()

