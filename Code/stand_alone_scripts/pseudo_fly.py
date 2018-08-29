#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Carries out convolutional neural network analysis
- Option to carry out pseudo-simulation on the fly analysis
"""

__author__ = 'Lucrezia Lorenzon (EMAIL_HERE)'

import os
import shutil
import itertools
import sys
from PIL import Image
from math import sqrt
import numpy as np
import ntpath
import random 
import csv
import json
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams #Note: Thhe version number has to be 2.2.2
import matplotlib.pyplot as plt
import pydot
import graphviz
import tensorflow
import keras#Note: Thhe version number has to be 2.1.3
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils #Utilities that help in the data transformation
from keras import backend as K
from keras.backend import eval
from keras.utils import plot_model
from sklearn.utils import shuffle
from keras.models import load_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
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
###########CNN VISUALISATION##############
##########################################


def classification_report_csv(report):
	report_data = []
	lines = report.split('\n')
	for line in lines[2:-3]:
		row = {}
		row_data = line.split('      ')
		row['class'] = row_data[1]
		row['precision'] = float(row_data[2])
		row['recall'] = float(row_data[3])
		row['f1_score'] = float(row_data[4])
		row['support'] = float(row_data[5])
		report_data.append(row)
	dataframe = pd.DataFrame.from_dict(report_data)
	dataframe.to_csv(CNN_dir +'/'+'classification_report.csv', index = False)
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


##########################################
##########ON THE FLY SIMULATIONS##########
##########################################

def load_model():
	model = model_from_json(open(CNN_dir +'/model.json').read())
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.load_weights(CNN_dir + '/weights.h5')
	return model


def save_model(model):    
	json_string = model.to_json()
	open(CNN_dir +'/model.json', 'w').write(json_string)
	model.save_weights(CNN_dir + '/weights.h5', overwrite=True)

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

##########################################
##############CNN PARAMETERS##############
##########################################
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

##########################################
##########METHOD OF TRAINING##############
##########################################
print width_screen * "-" 
print('METHOD OF TRAINING'.center(width_screen)) 
print width_screen * "-" 

#Setting the method of training
training_loop = True
while training_loop: 
	training_option_1, training_option_2 = options_menu("Training Methods", ['Reuse data each epoch','Pseudo-simulation on the fly'])
	trianing_options = input("Enter your choice [1-2]: ")
	if trianing_options ==1:
		print width_screen * "-"     
		print "Choice: {} has been selected".format(training_option_1)
		training = 1
		training_loop = False
	elif trianing_options ==2: 
		print width_screen * "-"     
		print "Choice: {} has been selected".format(training_option_2)
		training = 2
		training_loop = False
	else:
		print("Wrong option selection. Enter your choice again")



########################################################################
#########################GET NAME TO SAVE MODEL#########################
########################################################################

order = 'Write the name in which to save the model'
userInput = str(raw_input(order))
model_file_name = os.path.splitext(userInput)[0]

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

#If re-sized images do not already exist: resize
list_folders = next(os.walk(simulation_images))[1]
for folder in tqdm.tqdm(list_folders,total = len(list_folders), unit = 'folders'):
	if not os.path.isdir(simulation_images + '/' + folder + "/resize_col_" + str(col_size) + "_row_" + str(row_size)):
		resize_image(simulation_images + '/' + folder, col_size, row_size)


#####################################################
#####CREATING DIRECTORIES FOR CNN RESULTS############
#####################################################
#Create directory for the CNN
CNN_dir = results_directory + '/CNN'
#Create a Directory for training images: if does not already exist
trianing_dir = CNN_dir + '/training_data'
if os.path.exists(trianing_dir):
	shutil.rmtree(trianing_dir)
os.makedirs(trianing_dir)
#Create a Directory for testing images: if does not already exist
testing_dir = CNN_dir + '/testing_data'
if os.path.exists(testing_dir):
	shutil.rmtree(testing_dir)
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
		list_images = listdir_nohidden(image_roots)
		total_images.extend(list_images)
	
	#Randomly shuffle the order of images in the list
	random.seed(23)
	random.shuffle(total_images) 
	no_image_class = len(total_images)
	test_split = int(float(no_image_class)/float(100)*float(data_split[2]))
	train_val_split = no_image_class - test_split
	test = [total_images[i] for i in range(train_val_split,no_image_class )]
	train = [total_images[i] for i in range(train_val_split)]
	for image in train:
		shutil.copy(image, trianing_dir + '/' + co_efficient + '/' +  ntpath.basename(image))
	for image in test: 
		shutil.copy(image, testing_dir + '/' + co_efficient + '/' +  ntpath.basename(image))

################################################
###CREATING IM_INDEX: FOR TRAINING DIR IMAGES###
################################################

#create list of files contained in each selection co-efficient folder
listing = listdir_nohidden(trianing_dir)
selection_coefficients_training_data = list()
training_data = list()
full_path_training_data = list()
label_info = list()
for item in listing: 
		root = item
		root_contents = listdir_nohidden(root)
		if root_contents:
			selection_coefficients_training_data.append(os.path.basename(os.path.normpath(root)))
			label_info.append(len(listdir_nohidden(root)))
			full_path_training_data.append(root_contents)

num_selection_coefficients = len(full_path_training_data)
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
	open_image = np.asarray(PIL.Image.open(image)).flatten()
	open_image = open_image.astype('float32')
	im_matrix[index,:] = open_image
	index += 1

#Labeling of images: basically labeling by the selection coefficient place in list( note in the practice run we only have one selection co-efficeint so the labeling list is full of zeros)
label = np.zeros((num_samples,),dtype=int)
start = 0
stop = label_info[0]
for i in range(num_selection_coefficients):
	label[start:stop] = i
	start = start + label_info[i]
	stop = stop + label_info[i]

#Create a dictionary where the label corresponds to the actual selection co_efficient label
#convert the selection co-efficient to list
selection_coefficients_training_data = map(int, selection_coefficients_training_data)
class_label_dict = dict(zip(np.unique(label), selection_coefficients_training_data))

################################################
#################FINAL SHUFFLE##################
################################################
#Shuffle the matrices: the two matrices still correctly correspond with each other
im_matrix,label = sklearn.utils.shuffle(im_matrix,label,random_state=2)


############################################################################
################CONVOLUTIONAL NEURAL NETWORK################################
############################################################################
############################################################################
############################CNN ARCHITECTURE################################
############################################################################
#Get time per epoch
class TimeHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.times = []

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)


################################################
########PARAMETERS USED IN ARCHITECTURE#########
################################################
print width_screen * "-" 
print('CNN PARAMETERS'.center(width_screen)) 
print width_screen * "-" 
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
from keras.layers import LeakyReLU

#This model uses a linear sequential format
CNN = keras.models.Sequential()

#ROUND ONE OF PATTERN: CONV, MAX POOL, DROPOUT
#Declare the input layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1),padding='same', data_format='channels_last', input_shape=(row_size,col_size,channels)))
CNN.add(LeakyReLU(alpha=0.1))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.3))

#ROUND TWO OF PATTERN: CONV, MAX POOL, DROPOUT
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1),padding='same', data_format='channels_last'))
CNN.add(LeakyReLU(alpha=0.1))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ROUND THREE OF PATTERN: CONV, MAX POOL, DROPOUT
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1),padding='same', data_format='channels_last'))
CNN.add(LeakyReLU(alpha=0.1))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.7))

#ADDING A FULLY CONNECTED LATER
#Weights of previous layers are flattened(made 1D) before passing to the fully connected dense layer
CNN.add(keras.layers.core.Flatten())
CNN.add(keras.layers.core.Dense(128))
CNN.add(LeakyReLU(alpha=0.1))
#Add dropout layer: regularises model to prevent over-fitting
#CNN.add(keras.layers.core.Dropout(rate=0.5))
#Add output layer
CNN.add(keras.layers.core.Dense(nb_classes, activation='softmax'))
#Summary of the CNN architecture
CNN.summary()
#IMAGE OF THE MODEL
keras.utils.plot_model(CNN, to_file=CNN_dir +'/'+'model.png')

#IMAGE OF THE MODEL
#keras.utils.plot_model(CNN, to_file=CNN_dir +'/'+'model.png',show_shapes = True, show_layer_names = True, rankdir='LR')


#######################################################
########CONVOLUTIONAL NEURAL NETWORK: COMPILE##########
#######################################################
#Compiling the neural network using KL divergence as a loss function. 
#Adams optimizer is still used
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##########################################################################
############################TRAINING THE MODEL############################
##########################################################################

if training == 1: 
	#Test_size: The proportion of files in the training im_matrix to use for training evaluation
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
	nb_classes = num_selection_coefficients
	y_train= keras.utils.np_utils.to_categorical(y_train,nb_classes)
	y_val= keras.utils.np_utils.to_categorical(y_val,nb_classes)

	#Saving the split testing and training data to the CNN Directory
	np.save(CNN_dir + '/' + model_file_name + "_"  + "X_train", X_train,allow_pickle=False)
	np.save(CNN_dir + '/' + model_file_name +  "_"  + "X_val", X_val,allow_pickle=False)
	np.save(CNN_dir + '/' + model_file_name  +  "_"  + "y_train", y_train,allow_pickle=False)
	np.save(CNN_dir + '/' + model_file_name  +  "_"  + "y_val", y_val,allow_pickle=False)
	del im_matrix, label

	#Train the model: resuses the data at each epoch
	time_callback = TimeHistory()
	#Fitting the model as before
	hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, epochs=epochs, verbose=1, validation_data=(X_val, y_val),callbacks=[time_callback])
	times = time_callback.times
	time_dataframe = pd.DataFrame({'Epoch_time':times})
	time_dataframe.to_csv(CNN_dir +'/'+ model_file_name + "_" + "epoch_time.csv")


	#Fitting the model as before
if training == 2:

	#convert class vectors to binary class matrices
	nb_classes = len(selection_coefficients)
	label_binary_class= np_utils.to_categorical(label,nb_classes)
	################################################
	###SPLITING TRAINING DATA FOR SIM-ON-THE-FLY####
	################################################
	#Spliting data and labels for pseudo-simulation-on-the-fly
	pseudo_simulation_list_data = np.array_split(im_matrix,epochs)
	pseudo_simulation_list_labels = np.array_split(label_binary_class,epochs)
	#delete the im_matrix adn label as no longer required for the program
	del im_matrix, label,label_binary_class
	#val_size: The proportion of files in the training im_matrix to use for training evaluation
	val_size = float(data_split[1])/float(data_split[0]+data_split[1])
	#######################################################
	########CONVOLUTIONAL NEURAL NETWORK:TRAINING##########
	#######################################################
	#Loop throught the training data: one epoch per training data split
	history = list()
	for i in range(len(pseudo_simulation_list_data)):
		#split the data 
		X_train, X_val, y_train, y_val = train_test_split(pseudo_simulation_list_data[i], pseudo_simulation_list_labels[i], test_size=val_size, random_state=4)
		#Format the training data
		X_train = X_train.reshape(X_train.shape[0], row_size, col_size, channels)
		X_val = X_val.reshape(X_val.shape[0], row_size, col_size, channels)
		#normalize the training data 
		X_train /= 255
		X_val /= 255
		#Set the counters to allow learning rate decay to be tracked
		initial_epoch_counter = i +1
		epoch_number_counter = i +2 

			#Train the model: resuses the data at each epoch
	

		#fit the model using the unqiue training data
		if i == 0: 
		#The first iterations does not require weights to be loaded: since the first 'epoch'
			time_callback = TimeHistory()
			hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, nb_epoch=initial_epoch_counter, verbose=1, validation_data=(X_val, y_val),callbacks=[time_callback])
			times = time_callback.times
			time_dataframe = pd.DataFrame({'Epoch_time':times})
			time_dataframe.to_csv(CNN_dir +'/'+ model_file_name + str(i) +"_" + "epoch_time.csv")
			#Save the weights at each epoch
			CNN.save_weights(CNN_dir + '/weights.h5',overwrite=True)
			#Save the data used for training this "epoch"
			print(eval(CNN.optimizer.iterations))
			np.save(trianing_dir+ '/' + "X_train_" + str(initial_epoch_counter), X_train, allow_pickle=False)
			np.save(trianing_dir+ '/' + "X_val_" + str(initial_epoch_counter), X_val, allow_pickle=False)
			np.save(trianing_dir+ '/' + "y_train_" + str(initial_epoch_counter), y_train, allow_pickle=False)
			np.save(trianing_dir+ '/' + "y_val_" + str(initial_epoch_counter), y_val, allow_pickle=False)
			history.append(hist.history)
			print(eval(CNN.optimizer.lr))
		if i > 0:
		#All iterations after the first require weights from previous model epochs to be loaded.
			CNN.load_weights(CNN_dir + '/weights.h5')
			#Fit the model from the preloaded weights
			time_callback = TimeHistory()
			hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, nb_epoch = initial_epoch_counter, verbose = 1, validation_data = (X_val, y_val),initial_epoch = i ,callbacks=[time_callback])
			times = time_callback.times
			time_dataframe = pd.DataFrame({'Epoch_time':times})
			time_dataframe.to_csv(CNN_dir +'/'+ model_file_name + str(i) +"_" + "epoch_time.csv")
			print(eval(CNN.optimizer.iterations))
			#Save the weights at each epoch
			CNN.save_weights(CNN_dir + '/weights.h5',overwrite=True)
			print(eval(CNN.optimizer.lr))
			#Save the data used for training this "epoch"
			np.save(trianing_dir+ '/' + "X_train_" + str(initial_epoch_counter), X_train, allow_pickle=False)
			np.save(trianing_dir+ '/' + "X_val_" + str(initial_epoch_counter), X_val, allow_pickle=False)
			np.save(trianing_dir+ '/' + "y_train_" + str(initial_epoch_counter), y_train, allow_pickle=False)
			np.save(trianing_dir+ '/' + "y_val_" + str(initial_epoch_counter), y_val, allow_pickle=False)
			history.append(hist.history)

history = {k: [dic[k][0] for dic in history] for k in history[0]}

#######################################################
########CNNL LOSSES AND ACCURACIES: PLOTS##############
#######################################################
if training == 1:

	golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
	fig, ax = plt.subplots(figsize=golden_size(6))
	#Create dataframe of history object
	hist_df = pd.DataFrame(hist.history)
	#Plot the dataframe
	hist_df.plot(ax=ax)
	#Set the x and y labels
	ax.set_ylabel('Accuracy/Loss')
	ax.set_xlabel('# epochs')
	#ax.set_ylim(.99*hist_df[1:].values.min(), 1.1*hist_df[1:].values.max())
	plt.savefig(CNN_dir +'/' + model_file_name + "_loss_acc.pdf")

	hist_df.to_csv(CNN_dir +'/'+ model_file_name + "_" + "Training_History.csv")

if training == 2:
	train_loss=history['loss']
	val_loss=history['val_loss']
	train_acc=history['acc']
	val_acc=history['val_acc']
	hist_df = pd.DataFrame({'train_loss': train_loss,
     'val_loss': val_loss,
     'train_acc': train_acc,
     'val_acc': val_acc
    })
	hist_df.to_csv(CNN_dir +'/'+ model_file_name + "_" + "Training_History.csv")

#######################################################
######CONVOLUTIONAL NEURAL NETWORK: VALIDATION#########
#######################################################
#create list of files contained in each selection co-efficient folder
listing = listdir_nohidden(testing_dir)

#Count the number of images of each class: this takes into accoutnt that some classes fail to have entries in the testing data
selection_coefficients_testing_data = list()
full_path_testing_data = list()
label_info = list()
for item in listing: 
		root = item
		root_contents = listdir_nohidden(root)
		selection_coefficients_testing_data.append(os.path.basename(os.path.normpath(root)))
		label_info.append(len(listdir_nohidden(root)))
		full_path_testing_data.append(root_contents)


#Flatten the list of Full paths: this ignores any lists that might have been empty
flat_full_path_testing_data = [item for sublist in full_path_testing_data for item in sublist]

#Create matrix: contains all the images
num_samples = len(flat_full_path_testing_data)
im_matrix_rows = len(flat_full_path_testing_data)
im_matrix_columns = row_size*col_size
im_matrix = np.empty((im_matrix_rows,im_matrix_columns), dtype='float32')
index = 0

#Flattening of images and creation of im_matrix,
for image in flat_full_path_testing_data : 
	open_image = np.asarray(PIL.Image.open(image)).flatten()
	open_image = open_image.astype('float32')
	im_matrix[index,:] = open_image
	index += 1

#Labeling of images: basically labeling by the selection coefficient place in list: this takes into account any selection co-efficient classes that might have been empty
label = np.zeros((num_samples,),dtype=int)
start = 0
stop = label_info[0]
for i in range(num_selection_coefficients):
	label[start:stop] = i
	start = start + label_info[i]
	stop = stop + label_info[i]

#Shuffle the matrices: the two matrices still correctly correspond with each other
im_matrix,label = sklearn.utils.shuffle(im_matrix,label,random_state=2)

X_test_data = im_matrix.reshape(im_matrix.shape[0], row_size, col_size, channels)
#Normalise the data
X_test_data /= 255
#convert class vectors to binary class matrices
nb_classes = num_selection_coefficients
y_test_data= keras.utils.np_utils.to_categorical(label,nb_classes)
#Save the testing and training data
np.save(CNN_dir + '/' + "x_test" + "_" +  model_file_name ,X_test_data,allow_pickle=False)
np.save(CNN_dir + '/' + "y_test" + "_" +  model_file_name ,y_test_data,allow_pickle=False)

#Evaluate the model using unseen testing data
score = CNN.evaluate(X_test_data ,y_test_data, batch_size = None, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

########################################################################
########################SAVE TRAINING HISTORY###########################
########################################################################
hist_df.to_csv(CNN_dir +'/'+ model_file_name + "_" + "Training_History.csv")
########################################################################
############################SAVE MODEL EVALUATION#######################
########################################################################
score_df = {'test_loss' : score[0], 'test-accuarcy' : score[1]}
#Save the class_label association
with open(CNN_dir +'/'+ model_file_name + '_eval_score.json', 'w') as fp:
	json.dump(score_df, fp)


# ############################################################################
# ################################CNN PLOTS###################################
# ############################################################################


# ########################################################################
# ###########################FILTER WEIGHTS###############################
# ########################################################################
# #Plot weights of first convo layer
# plot_weights(0,filters,6,6)
# plt.clf()
# plt.cla()
# plt.close()
# #Plot weights of second convo layer
# plot_weights(3,filters,6,6)
# plt.clf()
# plt.cla()
# plt.close()
# #Plot weights of third convo layer
# plot_weights(6,filters,6,6)
# plt.clf()
# plt.cla()
# plt.close()

# #######################################################################
# ######VISUALISE THE FIRST LAYER ON CONVOLUTIONS ON INPUT IMAGE#########
# #######################################################################

# #Choose input image + save it in the CNN directory
# input_image = X_train[1:2,:,:,:]
# fig = plt.figure()
# plt.imshow(input_image[0,:,:,0],cmap='gray')
# plt.savefig(CNN_dir +'/'+ "training_image.pdf",bbox_inches='tight')



# layer_outputs = [layer.output for layer in CNN.layers]
# activation_model = Model(inputs=CNN.input, outputs=layer_outputs)
# activations = activation_model.predict(input_image)

# for i in range(9):
# 	display_activation(activations,6,6,i,filters)
# 	plt.clf()
# 	plt.cla()
# 	plt.close()


# for i in range(9):
# 	display_activation_col(activations,6,6,i,filters)
# 	plt.clf()
# 	plt.cla()
# 	plt.close()

########################################################################
###########################CONFUSION MATRIX#############################
########################################################################

##########################
#####CONFUSION MATRIX#####
##########################

#Get predictions using the testing data: this data we know the true label
Y_pred = CNN.predict(X_test_data,batch_size=None, verbose=1)
#Convert to class labels
y_pred = np.argmax(Y_pred, axis=1)
#The true labels are contained within the y_test_data array: we need to convert to class label
y_truth = np.argmax(y_test_data, axis=1)
#Get the confusion matrix: Note: the number of labels is dependent on the number of classes in the testing data
cm = confusion_matrix(y_truth,y_pred,labels = class_label_dict.keys())
#Create a dataframe of the confusion matrix 
df_cm = pd.DataFrame(cm,index= np.arange(len(class_label_dict.values())))
np.savetxt(CNN_dir +'/'+ model_file_name + "_" + "y_pred.csv", y_pred , delimiter=",")
np.savetxt(CNN_dir +'/'+ model_file_name + "_" + "y_truth.csv", y_truth , delimiter=",")
##########################
##DATAFRAME MANIPULATION##
##########################
#class_label_dict is a dictionary that contains the selection co-efficient to class associations

#Sort the dictionary based on the class labels
sorted_x = sorted(class_label_dict.items(), key=operator.itemgetter(1))
#List of the sorted index values
index_sorted = [x[0] for x in sorted_x]
#List of the sorted selection co-efficients
sel_sorted = [x[1] for x in sorted_x]
#Rearrange the database columns based on the ordered tuple from dict
df = df_cm[index_sorted]
#Rearrange the database rows based on the ordered tuple from dict
df = df.reindex(index_sorted)

##########################
####PLOT THE DATAFRAME####
##########################

#Plot the confusion matrix
plt.figure(figsize = (10,7))
#Centralise the tick marks
tick_marks = np.arange(len(class_label_dict.values()))
cen_tick = tick_marks + 0.5
#Plot the dataframe 
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True,annot_kws={"size": 12})
#Add the labels and the postions of the tick marks
plt.xticks(cen_tick, sel_sorted, rotation=45, fontsize=8)
plt.yticks(cen_tick, sel_sorted,rotation=45, fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
#Save the plot
plt.savefig(CNN_dir +'/'+ model_file_name + "_" + "Confusion Matrix.pdf",bbox_inches='tight')

#######################################
####Save Confusion matrix dataframe####
#######################################
df_cm.to_csv(CNN_dir +'/'+ model_file_name + "_" + "confusion_matrix_data.csv")

########################################################################
##########################CLASSIFICATION REPORT#########################
########################################################################
selection_coefficients_str = map(str, class_label_dict.values())
#This could be formatted to be in a given order
cr = classification_report(y_truth, y_pred, labels= class_label_dict.keys(), target_names = selection_coefficients_str )
classification_report_csv(cr)

#########################################################################
##############################ROC/AUC CURVES#############################
#########################################################################


#Get the predictions using the X_test_data
y_score = CNN.predict(X_test_data)

# Compute ROC curve and ROC area for each class
fpr = dict()#This is the false positive rate dict
tpr = dict()#THis is the true positive rate dict
roc_auc = dict()#This stores the ROC-AUC data dict

#For each class calculate the fpr, tpr and auc data: store in the initialised dictionaries
for i in range(nb_classes):
	fpr[i], tpr[i], _ = roc_curve(y_test_data[:, i], y_score[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_data.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
	mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# #Save each of the dictionaries: will be used in analysis of performance of each CNN
# #Save the class_label association
# np.save(CNN_dir +'/'+ model_file_name + '_fpr.npy',fpr)
#  # data2=np.load(CNN_dir +'/'+ model_file_name + '_fpr.npy')
#  # data2[()]
# np.save(CNN_dir +'/'+ model_file_name + '_tpr.npy',tpr)
# np.save(CNN_dir +'/'+ model_file_name + '_roc_auc.npy',roc_auc)


fpr_dataframe = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in fpr.iteritems() ]))
tpr_dataframe = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tpr.iteritems() ]))
roc_auc_dataframe = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in roc_auc.iteritems() ]))

fpr_dataframe.to_csv(CNN_dir +'/'+ model_file_name + "_" + "fpr_dataframe.csv")

tpr_dataframe.to_csv(CNN_dir +'/'+ model_file_name + "_" + "tpr_dataframe.csv")

roc_auc_dataframe.to_csv(CNN_dir +'/'+ model_file_name + "_" + "roc_auc_dataframe.csv")


"""
checking that the weights are loaded correctly 

#Weights of model initially
w1 = model.get_weights()
#Save the weights: overwrite = True is important ensures that the files is re-written each iteration 
model.save_weights( ,overwrite=True)
#Load the model weights: do not fit the model or change it again
model.load_weights('your_saved_weights.h5')
#Weights of the model after fitting
w2 = model.get_weights()

#Checking that the model weights are the same 
for a,b in zip(w1, w2):
  if np.all(a == b):
	print "wtf is happening"



model.optimizer.get_config()
model.loss.items()
"""










