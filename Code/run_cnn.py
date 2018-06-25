#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Carries out convolutional neural network analysis
"""

__author__ = 'Lucrezia Lorenzon (EMAIL_HERE)'

import os
import shutil
import itertools
from PIL import Image
from math import sqrt
import numpy as np
import ntpath
import random 
import csv
import json
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams #Note: Thhe version number of numpy has to be 2.2.2
import matplotlib.pyplot as plt
import pydot
import graphviz
import tensorflow
import keras#Note: Thhe version number of numpy has to be 2.1.3
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils #Utilities that help in the data transformation
from keras import backend as K
from keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
#Getting the terminal screen size: used when formatting user input
import subprocess
length_screen, width_screen = subprocess.check_output(['stty', 'size']).split()
width_screen  = int(width_screen)
#Packages required for progress bar
from tqdm import tqdm, trange
from time import sleep
rcParams.update({'figure.autolayout': True})
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last') #e' quello giusto per Theano
print("Image_data_format is " + K.image_data_format())

##################################################################################################
##################################################################################################
##########################################FUNCTIONS###############################################
##################################################################################################
##################################################################################################

##########################################
#########FUNCTIONS FOR USER INPUT#########
##########################################

def getfile(filetype):
	"""
	Checks if the input file/folder exists: if not asks for new file path until correct one found
	
	Keyword Arguments: 
		filetype (string) -- type of file looking for, eg 'reference genome'
	
	Returns: 
		user_input (string) -- path to specifed files

	"""
	loop = True
	order = 'Path to the %s: ' %(filetype)
	error = 'Path entered does not exist. Enter the path again"'
	while loop: 
		user_input = raw_input(order)
		user_input = user_input.strip()
		result = os.path.exists(user_input)
		if result == True: 
			loop = False
			file_loc = user_input
		
		else: 
			print error
	return file_loc

def inputNumber(parameter_name):
	"""
	Function to get threshold value and check if it is within user specified limits
	
	Keyword Arguments: 
		parameter_name (string) -- name of parameter being specified
	
	Returns: 
		value (int) -- user specified paramter value

	"""
	order = 'Write the %s paramter value as a positive integer: ' %(parameter_name)
	while True:
		try:
			userInput = int(raw_input(order))       
			if userInput < 0:
				print("Not a positive integer! Try again.")
				continue
		except ValueError:
			print("Write postive integer in numerical form! Try again.")
			continue			
		else:
			return userInput 
			break 

def options_menu(title, options):

	"""
	Creates menus of user specified options

	Keyword Arguments: 
		title (string) -- Name of menu
		options (string) -- Specified user options

	Returns: 
		options (string) -- Specified user options
	"""
	print width_screen * "-"
	print(title.center(width_screen))
#	print '{:^{width_screen}}'.format(title,width_screen)
	print width_screen * "-"
	for x in range(len(options)):
		print str(x+1) + ". {}".format(options[x])
	print width_screen * "-"
	return(options)

##########################################
###########IMAGE PREPERATION##############
##########################################

def resize_image(parent_directory, img_columns,img_rows):
	"""
	Function to resize images
	
	Keyword Arguments: 
		parent_directory (string) -- directory contaiing images to be resized
	"""
	#List of img files in the directory for the specific simulation file
	listing = os.listdir(parent_directory)
	#Number of img files in the directory for the specific simulation file
	num_samples = len(listing)
	#Create Directory to store the re-sized images
	dir_nam =   parent_directory + '/resize_col_'  + str(img_columns) + '_row_' + str(img_rows)
	if not os.path.exists(dir_nam):
		os.makedirs(dir_nam)
	#Open image and resize --> (img_rows,img_columns)
	for file in listing:
		if file.endswith('.bmp'):
			im = Image.open(parent_directory + '/' + file)
			im_resized = im.resize((img_columns,img_rows))
			im_resized.save(dir_nam + '/' + file)
	return; 
##################################################################################################
##################################################################################################
#############################################MAIN#################################################
##################################################################################################
##################################################################################################

############################################################################
###########################DATA INITIALISATION##############################
############################################################################

##########################################
################GET FILE PATHS############
##########################################
print width_screen * "-" 
print('FILE PATHS'.center(width_screen)) 
print width_screen * "-" 
#Get the location of the simulations 
simulation_files = getfile('Simulation Files')
#Get the location of the results directory
results_directory = getfile('Results Directory')
#Using the path to the results directory we get the path to the simulation images
simulation_images = results_directory + '/simulation_data'
print width_screen * "-" 

##########################################
###########DATA_SET INFORMATION###########
##########################################
'''
NOTE: We assume that the values of NITER, NCHROMS and NREF are constant between the simulation files
'''
#Open the index file
with open(simulation_files + '/data.json', 'r') as fp:
	data_index = json.load(fp)
#Determine which simulation files are currently in ananlysis: Deleted simulations are still kept in index file [marked as 'inactive']
active_dict = filter(lambda x: x.get('active') == 'active', data_index)
#Get set of the unqiue selection_coefficients from those active simulation files
selection_coefficients = set([d['SAA'] for d in active_dict if 'SAA' in d])
NITER = 300 #For the sake of this thing else: int(active_dict[0]['NITER'])
NCHROMS = int(active_dict[0]['NCHROM'])
NREF = int(active_dict[0]['NREF'])

##########################################
###########DETERMINE IMAGE SIZE###########
##########################################
#The image size is determined by: Channels, width and height
print width_screen * "-" 
print('DETERMINING IMAGE SIZE'.center(width_screen)) 
print width_screen * "-" 
#Setting the number of channels
channel_loop = True
while channel_loop: 
	channel_option_1, channel_option_2 = options_menu("Channel Number", ['Black & White Images [1 Channel]','RGB Images [3 Channels]'])
	channel_options = input("Enter your choice [1-2]: ")
	if channel_options ==1:
		print width_screen * "-"     
		print "Choice: {} has been selected".format(channel_option_1)
		channels = 1
		channel_loop = False
	elif channel_options ==2: 
		print width_screen * "-"     
		print "Choice: {} has been selected".format(channel_option_2)
		channels = 3
		channel_loop = False
	else:
		print("Wrong option selection. Enter your choice again")
#Setting the width and height of the images
Resize_loop_01 =True      
while Resize_loop_01:
	option_1, option_2,option_3, option_4, option_5,option_6 = options_menu("Resize Options", ['Raw Image Dimensions','Square 32 x 32','Square 64 x 64','Square 96 x 96','Square 128 x 128','Square 256 x 256'])
	resize_options = input("Enter your choice [1-6]: ")
	if resize_options==1:
		print width_screen * "-"     
		print "Choice: {} has been selected".format(option_1)
		Resize_loop_01=False 
		Resize_loop_02 =True
		#Open the file containing the dimension data
		with open(simulation_images + '/img_dimension.json', 'r') as fp:
			dimensions = json.load(fp)
		row_size = NCHROMS
		while Resize_loop_02:
			raw_option_1, raw_option_2, raw_option_3 = options_menu("Raw Dimension Options", ['Mean','Max','Min'])
			raw_resize_options = input("Enter your choice [1-3]: ")
			if raw_resize_options == 1:
				print width_screen * "-" 
				print "Choice: {} has been selected".format(raw_option_1)
				col_size= dimensions.get('mean')
				Resize_loop_02 =False	
			elif raw_resize_options == 2:
				print width_screen * "-" 
				print "Choice: {} has been selected".format(raw_option_2)
				col_size = dimensions.get('max')
				Resize_loop_02 =False
			elif raw_resize_options == 3:
				print width_screen * "-" 
				print "Choice: {} has been selected".format(raw_option_3)
				col_size = dimensions.get('min')
				Resize_loop_02 =False
			else:
				print("Wrong option selection. Enter your choice again")
	elif resize_options==2:
		print width_screen * "-"
		print "Choice: {} has been selected".format(option_2)
		col_size = 32
		row_size = 32
		Resize_loop_01=False 
	elif resize_options==3:
		print width_screen * "-"
		print "Choice: {} has been selected".format(option_3)
		col_size = 64
		row_size = 64
		Resize_loop_01=False 
	elif resize_options==4:
		print width_screen * "-"
		print "Choice: {} has been selected".format(option_4)
		col_size = 96
		row_size = 96
		Resize_loop_01=False 
	elif resize_options==5:
		print width_screen * "-"
		print "Choice: {} has been selected".format(option_5)
		col_size = 128
		row_size = 128
		Resize_loop_01=False 
	elif resize_options==6:
		print width_screen * "-"
		print "Choice: {} has been selected".format(option_6)
		col_size = 256
		row_size = 256
		Resize_loop_01=False 
	else:
		print("Wrong option selection. Enter your choice again")

############################################################################
############################DATA PREP#######################################
############################################################################


#####################################################
#################RE-SIZE THE IMAGES##################
#####################################################
#For each folder containing simulations: resize the images
#Moves the images to the training directory that was created above
print width_screen * "-" 
print('RESIZING IMAGES'.center(width_screen)) 
print width_screen * "-" 

list_folders = next(os.walk(simulation_images))[1]
for folder in tqdm(list_folders,total = len(list_folders), unit = 'folders'):
	if not folder.startswith('resize'):
		if not folder.startswith('.'):
			if not folder == 'CNN_41':
				if not folder == 'training_data':
					resize_image(simulation_images + '/' + folder, col_size, row_size)

#####################################################
#####CREATING DIRECTORIES FOR CNN RESULTS############
#####################################################
#Create directory for the CNN
CNN_dir = results_directory + '/CNN'
#Create a Directory for training images: if does not already exist
trianing_dir = CNN_dir + '/training_data'
if not os.path.exists(trianing_dir):
	os.makedirs(trianing_dir)
#Create a Directory for testing images: if does not already exist
testing_dir = CNN_dir + '/testing_data'
if not os.path.exists(testing_dir):
	os.makedirs(testing_dir)


#####################################################
###DEFINE DATA SPLIT: TRAINING:EVLUATION: TESTING####
#####################################################
#Define how the data is to be split(can be user defined)
data_split = [50,25,25]

#####################################################
###SPLIT DATA ACCORDING TO SELECTION CO_EFFICIENT####
#####################################################
#Get a list of files with each co-efficient values
for co_efficient in selection_coefficients:
	total_images = []
	if not os.path.exists(trianing_dir +'/' + co_efficient):
		os.makedirs(trianing_dir +'/' + co_efficient)
	if not os.path.exists(testing_dir +'/' + co_efficient):
		os.makedirs(testing_dir +'/' + co_efficient)
	files = []
	common_dict = filter(lambda x: x.get('SAA') == co_efficient, active_dict)
	for item in common_dict:
		files.append({k: common_dict[common_dict.index(item)].get(k, None) for k in ('name','SAA')})
	#Get just the file names
	names = sorted([d['name'] for d in files if 'name' in d])
	for file_name in names: 
		short_name = file_name.partition('.txt')[0]
		image_roots = simulation_images + '/' +  short_name + '/resize_col_'  + str(col_size) + '_row_' + str(row_size)
		list_images = os.listdir(image_roots)
		#Get list of all the images in selection co-efficient category
		list_image_path = [image_roots + '/' + s  for s in list_images]
		total_images.extend(list_image_path)
	#Randomly shuffle the order of images in the list
	random.seed(23)
	random.shuffle(total_images) 
	no_image_class = len(total_images)
	test_split = int(float(no_image_class)/float(100)*float(data_split[2]))
	train_val_split = no_image_class - test_split
	test = [total_images[i] for i in range(train_val_split,no_image_class )]
	train = [total_images[i] for i in range(train_val_split)]
	for image in train:
		if not image.startswith('.'): 
			shutil.copy(image, trianing_dir + '/' + co_efficient + '/' +  ntpath.basename(image))
	for image in test: 
		if not image.startswith('.'):
			shutil.copy(image, testing_dir + '/' + co_efficient + '/' +  ntpath.basename(image))

################################################
###CREATING IM_INDEX: FOR TRAINING DIR IMAGES###
################################################
#create list of files contained in each selection co-efficient folder
listing = os.listdir(trianing_dir)
num_selection_coefficients = len(listing) - 1
training_data = list()
full_path_training_data = list()
label_info = list()
for item in listing: 
	if not item.startswith('.'):
		root = trianing_dir + '/' + item
		root_contents = os.listdir(root)
		label_info.append(len(os.listdir(root)))
		full_path_root_contents = [root +'/' + x for x in root_contents if not x.startswith('.')]
		full_path_training_data.append(full_path_root_contents)
#Flatten the list of Full paths
flat_full_path_training_data = [item for sublist in full_path_training_data for item in sublist]
#Create matrix: contains all the images
num_samples = len(flat_full_path_training_data)
im_matrix_rows = len(flat_full_path_training_data)
im_matrix_columns = row_size*col_size
im_matrix = np.empty((im_matrix_rows,im_matrix_columns), dtype='float32')
index = 0
#Flattening of images and creation of im_matrix,
for image in flat_full_path_training_data: 
	open_image = np.asarray(Image.open(image)).flatten()
	open_image = open_image.astype('float32')
	im_matrix[index,:] = open_image
	index += 1
#Labeling of images: basically labeling by the selection coefficient place in list( note in the practice run we only have one selection co-efficeint so the labeling list is full of zeros)
label = np.zeros((num_samples,),dtype=int)
start = 0
stop = NITER
for i in range(len(selection_coefficients)):
	label[start:stop] = i
	start = start + label_info[i]
	stop = stop + label_info[i]
#Shuffle the files, so that the labels still correspond
im_matrix,label = shuffle(im_matrix,label,random_state=2)

################################################
######SPLIT TRAINING AND VALIDATION DATA########
################################################
#Test_size: The proportion of files in the training im_matrix to use for training evaluation
val_size = float(data_split[1])/float(data_split[0]+data_split[1])
X_train, X_val, y_train, y_val = train_test_split(im_matrix, label, test_size=val_size, random_state=4)

#X_train.shape[0]: The number of samples used for training
X_train = X_train.reshape(X_train.shape[0], row_size, col_size, channels)
X_val = X_val.reshape(X_val.shape[0], row_size, col_size, channels)

################################################
###########PREPROSESSING THE  DATA##############*
################################################
#Rescale the data: result = data vectors lie in the range of [0,1]
#As images lie on the range of [0,255], we carry out simple rescaling by dividing the data by 255.
X_train /= 255
X_val /= 255

#convert class vectors to binary class matrices
nb_classes = len(selection_coefficients)
y_train= np_utils.to_categorical(y_train,nb_classes)
y_val= np_utils.to_categorical(y_val,nb_classes)

#Saving the split testing and training data to the CNN41 folder
np.save(trianing_dir+ '/' + "X_train",X_train,allow_pickle=False)
np.save(trianing_dir+ '/' + "X_val",X_val,allow_pickle=False)
np.save(trianing_dir+ '/' + "y_train",y_train,allow_pickle=False)
np.save(trianing_dir+ '/' + "y_vak",y_val,allow_pickle=False)
del im_matrix, label

############################################################################
############################CNN ARCHITECTURE################################
############################################################################

################################################
########PARAMETERS USED IN ARCHITECTURE#########
################################################
print width_screen * "-" 
print('CNN PARAMETERS'.center(width_screen)) 
print width_screen * "-" 

#nb_classes = 41
######batch_size --> number of samples used per gradient update (default = 32)
#######epochs --> number of epochs to train the model; an epoch is an iteration over the entire x and y data provided
batch_size_parameter = inputNumber('Batch Size')
#batch_size_parameter = 32
print width_screen * "-" 
epochs = inputNumber('Epochs')
#epochs = 20
print width_screen * "-" 
filters = inputNumber('Filters')
#filters = 32
print width_screen * "-" 
kernel_size = inputNumber('Kernal Size')
#kernel_size = 3
print width_screen * "-" 
pooling_size = inputNumber('Pooling Size')
#pooling_size = 2
print width_screen * "-" 

#######################################################
########ARTIFICIAL NEURAL NETWORK ARCHTIECTURE#########
#######################################################
#This model uses a linear sequential format
CNN = Sequential()
#Declare the input layer
CNN.add(Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last', input_shape=(row_size,col_size,channels)))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add convolutional layer
CNN.add(Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last'))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(Dropout(rate=0.5))

#ADDING A FULLY CONNECTED LATER
#Weights of previous layers are flattened(made 1D) before passing to the fully connected dense layer
CNN.add(Flatten())
CNN.add(Dense(128, activation='relu'))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(Dropout(rate=0.5))
#Add output layer
CNN.add(Dense(nb_classes, activation='softmax'))
#Summary of the CNN architecture
CNN.summary()
#IMAGE OF THE MODEL
plot_model(CNN, to_file=CNN_dir +'/'+'model.png')

#######################################################
########CONVOLUTIONAL NEURAL NETWORK: COMPILE##########
#######################################################
#Loss function: cateogorical crossentropy
#Optimiser: adam
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#######################################################
########CONVOLUTIONAL NEURAL NETWORK:TRAINING##########
#######################################################
#verbose --> Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

#######################################################
######CONVOLUTIONAL NEURAL NETWORK: VALIDATION#########
#######################################################
#Create im_matrix of testing data
listing = os.listdir(testing_dir)
num_selection_coefficients = len(listing) - 1
testing_data = []
full_path_testing_data = []
label_info = []
for item in listing: 
	if not item.startswith('.'):
		root = testing_dir + '/' + item
		root_contents = os.listdir(root)
		label_info.append(len(os.listdir(root)))
		full_path_root_contents = [root +'/' + x for x in root_contents if not x.startswith('.')]
		full_path_testing_data.append(full_path_root_contents)

flat_full_path_testing_data = [item for sublist in full_path_testing_data for item in sublist]
#Create matrix: contains all the images
num_samples = len(flat_full_path_testing_data)
im_matrix_rows = len(flat_full_path_testing_data)
im_matrix_columns = row_size*col_size
im_matrix = np.empty((im_matrix_rows,im_matrix_columns), dtype='float32')
index = 0
#Flattening of images and creation of im_matrix,
for image in flat_full_path_testing_data: 
	open_image = np.asarray(Image.open(image)).flatten()
	open_image = open_image.astype('float32')
	im_matrix[index,:] = open_image
	index += 1
#Labeling of images: basically labeling by the selection coefficient place in list( note in the practice run we only have one selection co-efficeint so the labeling list is full of zeros)
label = np.zeros((num_samples,),dtype=int)
start = 0
stop = NITER
for i in range(len(selection_coefficients)):
	label[start:stop] = i
	start = start + label_info[i]
	stop = stop + label_info[i]
#Shuffle the files, so that the labels still correspond
im_matrix,label = shuffle(im_matrix,label,random_state=2)
X_test_data = im_matrix.reshape(im_matrix.shape[0], row_size, col_size, channels)
#Normalise the data
X_test_data /= 255
#convert class vectors to binary class matrices
nb_classes = len(selection_coefficients)
y_test_data= np_utils.to_categorical(label,nb_classes)
#Save the testing and training data
np.save(testing_dir + '/' + "x_test",X_test_data,allow_pickle=False)
np.save(testing_dir + '/' + "y_test",y_test_data,allow_pickle=False)

#Evaluate the model using unseen testing data
score = CNN.evaluate(X_test_data ,y_test_data, batch_size = None, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

############################################################################
################################CNN PLOTS###################################
############################################################################


#######################################################
########CNNL LOSSES AND ACCURACIES: PLOTS##############
#######################################################

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(epochs)
x_axis =np.zeros(len(xc))
for x,i in enumerate (xc):
	x_axis[i]=x+1

rcParams['axes.titlepad'] = 20
plt.figure(1,figsize=(7,5),facecolor='white')
plt.plot(x_axis,train_loss)
plt.plot(x_axis,val_loss)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training loss and validation loss',fontsize=12)
plt.grid(True)
plt.legend(['Training loss','Validation loss'],fontsize=12)
plt.style.use(['classic'])
plt.savefig(CNN_dir +'/'+ "Loss.eps")

plt.figure(2,figsize=(7,5),facecolor='white')
plt.plot(x_axis,train_acc)
plt.plot(x_axis,val_acc)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.title('Training accuracy and validation accuracy',fontsize=12)
plt.grid(True)
plt.legend(['Training accuracy','Validation accuracy'],fontsize=12,loc=4)
plt.style.use(['classic'])
plt.savefig(CNN_dir +'/'+ "Accuracy.eps")

#######################################################
##############CONFUSION MATRIX: PLOTS##################
#######################################################

Y_pred = CNN.predict(X_test_data,batch_size=None, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

#Convert the selection co-efficent set to a list
selection_coefficients_list = list(selection_coefficients)
classes = np.zeros(len(selection_coefficients))
for i in range(len(selection_coefficients)):
	classes[i] =  str(selection_coefficients_list[i])
classes = classes.astype('str')


cm = confusion_matrix(np.argmax(y_test_data,axis=1), y_pred)
np.set_printoptions(precision=2)
fig = plt.figure(facecolor='white')
title='Normalized confusion matrix'
cmap=plt.cm.Blues
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
#plt.colorbar(shrink=0.7)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
#plt.xticks(tick_marks, rotation=45, fontsize=6)
plt.yticks(tick_marks, classes, fontsize=8)
#plt.yticks(tick_marks)
#fmt = '.2f'
#thresh = cm.max() / 2.
#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, format(cm[i, j], fmt),
#             horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(CNN_dir +'/'+"confusion_matrix.eps",bbox_inches='tight')


#######################################################
##############CLASSIFICATION REPORT####################
#######################################################

cr = classification_report(np.argmax(y_test_data,axis=1),y_pred, target_names = classes)
print(cr)
np.save(CNN_dir +'/'+"classification_report",cr,allow_pickle=False)

#SAVE WHOLE MODEL (architecture + weights + training configuration[loss,optimizer] +
# state of the optimizer). This allows to resume training where we left off.
CNN.save(CNN_dir +'/'+"CNN_model.h5")
np.save(CNN_dir +'/'+"val_acc",val_acc,allow_pickle=False)
np.save(CNN_dir +'/'+"val_loss",val_loss,allow_pickle=False)
np.save(CNN_dir +'/'+"train_acc",train_acc,allow_pickle=False)
np.save(CNN_dir +'/'+"train_loss",train_loss,allow_pickle=False)
