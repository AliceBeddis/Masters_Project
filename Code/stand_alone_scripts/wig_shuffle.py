#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Adds noise in data to allow relationships between categories to be incorporated
- wiggle: re-labeling with neighbour categories
- norm: re-label based on a multinormal distribition where probabilities are drawn from a normal distribition
- kld: labels are decrete pmf. 
"""

__author__ = 'Alice Beddis (alice.beddis14@imperial.ac.uk)'

import glob
import os
import shutil
import itertools
import PIL
import math
import numpy as np
import ntpath
import random 
import csv
import json
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#from matplotlib import rcParams #Note: Thhe version number has to be 2.2.2
import pydot
import graphviz
import tensorflow
import keras#Note: Thhe version number has to be 2.1.3
from keras import backend as K
import sklearn
from sklearn.model_selection import train_test_split
import scipy.stats
import scipy.stats as ss
import pandas as pd
import subprocess
import tqdm
import time
import subprocess

################################################
matplotlib.rcParams.update({'figure.autolayout': True})
################################################
#Getting the terminal screen size: used when formatting user input
length_screen, width_screen = subprocess.check_output(['stty', 'size']).split()
width_screen  = int(width_screen)
################################################
keras.backend.set_image_dim_ordering('tf')
#K.set_image_dim_ordering('tf')
keras.backend.set_image_data_format('channels_last')
#K.set_image_data_format('channels_last') #e' quello giusto per Theano
print("Image_data_format is " + keras.backend.image_data_format())
################################################
#from random import *

##################################################################################################
##################################################################################################
##########################################FUNCTIONS###############################################
##################################################################################################
##################################################################################################

##########################################
#########FUNCTIONS FOR USER INPUT#########
##########################################

def listdir_nohidden(path):
	"""
	Gets list of files in a directory whilst ignoring hidden files
	
	Keyword Arguments: 
		path (string) -- path to directory in which to get list of files
	
	Returns: 
		 (string) -- list of directory contents with full paths

	"""
	return glob.glob(os.path.join(path, '*'))

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
			im = PIL.Image.open(parent_directory + '/' + file)
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
for folder in tqdm.tqdm(list_folders,total = len(list_folders), unit = 'folders'):
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
stop = NITER
for i in range(num_selection_coefficients):
	label[start:stop] = i
	start = start + label_info[i]
	stop = stop + label_info[i]

#Create a dictionary where the label corresponds to the actual selection co_efficient label
#convert the selection co-efficient to list
selection_coefficients_training_data = map(int, selection_coefficients_training_data)
class_label_dict = dict(zip(np.unique(label), selection_coefficients_training_data))


################################################################################################################################################
####################################################################THE WIGGLE##################################################################
################################################################################################################################################

#set the random seed: allows pseudoreplication
random.seed(34)

#Get list of unique labels, the index positions of when the first instance of the unique label is in the list[assumption: ordered list], count of unqiue labels 
unique,label_index_pos,label_counts = np.unique(label,return_index = True, return_counts = True)

#Delete the first position of the label_index_pos list as not used in spliting the label matrix
label_index_pos = np.delete(label_index_pos, 0) 

#Split the label matrix into arrays of unique label elements
split_label = np.split(label,label_index_pos)

for i in range(len(split_label)):
	#Per unique label, get the label subset and the count
	label_subset = split_label[i]
	label_counter = label_counts[i]
	#Different wiggle scenarios for end points (first label and last label) and the other labels
	if i == 0 or i == max(range(len(split_label))):
		#Only scramble if more than 1 image with this label
		if label_counter > 1:
			#Split for this instance is 0.25
			split = 0.25
			#Determine the number of label instances to scramble(note that int will truncate the float(we get a conservative number split value rather than rounding up))
			number_split = int(float(split) * float(label_counter))
			#Only if the namble of label instance to be scrambled is above  or equal to 1: scramble 
			if number_split >= 1:
				#randomly choose elements of lable list to re-label
				replace_these = np.random.choice(len(label_subset),number_split,replace = False)
				# convert to list: arrays cannot be used for indexing
				index_replacement = list(replace_these)
				if i == 0: 
					#For the first label, we re-lable with label class directly above
					number_replacement = i + 1
				if i == max(range(len(split_label))): 
					number_replacement = i - 1
					#For the last label, we re-lable with label class directly below
				#Re-label the subset
				label_subset[index_replacement] = number_replacement
	else: 
		#Only scramble if more than 1 image with this label
		if label_counter > 1:
			#The major split for this instance is 0.5
			first_split = 0.5
			#Determine the number of label instances to scramble(note that int will truncate the float(we get a conservative number split value rather than rounding up))
			number_first_split = int(first_split * label_counter)
			#Only if the namble of label instance to be scrambled is above  or equal to 1: scramble 
			if number_first_split > 1:
				#Minor split for this instance is 0.5. Ultimately end up with 50% original, 25% directly under and 25% directly over.
				second_split = 0.5
				#Determine the number of label instances to scramble(note that int will truncate the float(we get a conservative number split value rather than rounding up))
				number_second_split = int(float(second_split) * float(number_first_split))
				#Get index position of all elements to be replaced
				replace_these = np.random.choice(len(label_subset),number_first_split,replace = False)
				#Subset these index positions to those to be replaced with lower class and higher class
				replace_these_2 = np.random.choice(replace_these,number_second_split,replace = False)
				difference = np.setdiff1d(replace_these,replace_these_2)
				#Whether the index poistions are replaced with a higher class or a lower class is random
				options = [list(replace_these_2), list(difference)]
				lower_neighbour = options[np.random.choice(len(options),1,replace = False)[0]]
				upper_neighbour = list(np.setdiff1d(replace_these,lower_neighbour))
				#Re-label the subset accordingly
				label_subset[lower_neighbour] = i -1 
				label_subset[upper_neighbour] = i + 1
			if number_first_split == 1: 
				#if there is only one number to be replaced a different approach is taken
				#list of possible replacement labels.
				numbers = [i-1, i+1]
				#randomly chose the index in subset list to be replaced
				replace_these = np.random.choice(len(label_subset),number_first_split,replace = False)
				index_replacement = list(replace_these)
				#Randomly choose wether the index to be replaced is replaced by a higher or lower class label
				label_subset[index_replacement] = random.sample(numbers,  1)

################################################
#################FINAL SHUFFLE##################
################################################
#Shuffle the matrices: the two matrices still correctly correspond with each other
im_matrix,label = sklearn.utils.shuffle(im_matrix,label,random_state=2)

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

#Saving the split testing and training data to the CNN41 folder
np.save(trianing_dir+ '/' + "X_train",X_train,allow_pickle=False)
np.save(trianing_dir+ '/' + "X_val",X_val,allow_pickle=False)
np.save(trianing_dir+ '/' + "y_train",y_train,allow_pickle=False)
np.save(trianing_dir+ '/' + "y_val",y_val,allow_pickle=False)
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
CNN = keras.models.Sequential()

#ROUND ONE OF PATTERN: CONV, MAX POOL, DROPOUT
#Declare the input layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last', input_shape=(row_size,col_size,channels)))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ROUND TWO OF PATTERN: CONV, MAX POOL, DROPOUT
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last'))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ROUND THREE OF PATTERN: CONV, MAX POOL, DROPOUT
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last'))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ADDING A FULLY CONNECTED LATER
#Weights of previous layers are flattened(made 1D) before passing to the fully connected dense layer
CNN.add(keras.layers.core.Flatten())
CNN.add(keras.layers.core.Dense(128, activation='relu'))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))
#Add output layer
CNN.add(keras.layers.core.Dense(nb_classes, activation='softmax'))
#Summary of the CNN architecture
CNN.summary()
#IMAGE OF THE MODEL
keras.utils.plot_model(CNN, to_file=CNN_dir +'/'+'model.png')

#######################################################
########CONVOLUTIONAL NEURAL NETWORK: COMPILE##########
#######################################################
#Compiling the neural network using KL divergence as a loss function. 
#Adams optimizer is still used
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#######################################################
########CONVOLUTIONAL NEURAL NETWORK:TRAINING##########
#######################################################
#Fitting the model as before
hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))


######################################################
########CONVOLUTIONAL NEURAL NETWORK:PLOT TRAINING#####
#######################################################
#Set the proportion
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
plt.savefig(CNN_dir + "/loss_acc_cross_entropy.pdf")
################################################################################################################################################
###############################################################THE DESCRETE_NORMAL##############################################################
################################################################################################################################################

#re-label based on specific distribution
random.seed(34)

#Get list of unique labels, the index positions of when the first instance of the unique label is in the list[assumption: ordered list], count of unqiue labels 
unique,label_index_pos,label_counts = np.unique(label,return_index = True, return_counts = True)

#Delete the first position of the label_index_pos list as not used in spliting the label matrix
label_index_pos = np.delete(label_index_pos, 0) 

#Split the label matrix into arrays of unique label elements
split_label = np.split(label,label_index_pos)

#Initialise list that stores the probability distribution per label class
integer_normal = list()

#Initialise list that stores the re-sampled labels, based on the probability distribution
labels_prob = list()

#For each unique label: create normal distribution of probabilitys of re-sampling and carry out resampling
for i in range(len(split_label)):
	label_subset = split_label[i]
	label_counter = label_counts[i]
	#Create pmf of probabilities of re-sampling
	x = np.arange(min(unique), max(unique) + 1)
	xU, xL = x + 0.5, x - 0.5 
	prob = scipy.stats.norm.cdf(xU, loc = i, scale = 1) - ss.norm.cdf(xL, loc = i, scale = 1)
	#normalize the probabilities so their sum is 1
	prob = prob / prob.sum() 
	#Append the probabilities to list
	labels_prob.append(prob)
	#Re-sample based on the probabilities
	nums = np.random.choice(x, size = label_counts[i], p = prob)
	#Append re-sampled label sublists to list
	integer_normal.append(nums)
	normal_label = np.concatenate(integer_normal).ravel()

################################################
#################FINAL SHUFFLE##################
################################################
#Shuffle the matrices: the two matrices still correctly correspond with each other
im_matrix, normal_label  = sklearn.utils.shuffle(im_matrix,normal_label ,random_state=2)


################################################
######SPLIT TRAINING AND VALIDATION DATA########
################################################
#Test_size: The proportion of files in the training im_matrix to use for training evaluation
val_size = float(data_split[1])/float(data_split[0]+data_split[1])
X_train, X_val, y_train, y_val = train_test_split(im_matrix, normal_label , test_size=val_size, random_state=4)

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

#Saving the split testing and training data to the CNN41 folder
np.save(trianing_dir+ '/' + "X_train",X_train,allow_pickle=False)
np.save(trianing_dir+ '/' + "X_val",X_val,allow_pickle=False)
np.save(trianing_dir+ '/' + "y_train",y_train,allow_pickle=False)
np.save(trianing_dir+ '/' + "y_vak",y_val,allow_pickle=False)


############################################################################
############################CNN ARCHITECTURE################################
############################################################################

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
#This model uses a linear sequential format
CNN = keras.models.Sequential()
#Declare the input layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last', input_shape=(row_size,col_size,channels)))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last'))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))
#ADDING A FULLY CONNECTED LATER
#Weights of previous layers are flattened(made 1D) before passing to the fully connected dense layer
CNN.add(keras.layers.core.Flatten())
CNN.add(keras.layers.core.Dense(128, activation='relu'))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))
#Add output layer
CNN.add(keras.layers.core.Dense(nb_classes, activation='softmax'))
#Summary of the CNN architecture
CNN.summary()
#IMAGE OF THE MODEL
keras.utils.plot_model(CNN, to_file=CNN_dir +'/'+'model.png')

#######################################################
########CONVOLUTIONAL NEURAL NETWORK: COMPILE##########
#######################################################
#Compiling the neural network using KL divergence as a loss function. 
#Adams optimizer is still used
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#######################################################
########CONVOLUTIONAL NEURAL NETWORK:TRAINING##########
#######################################################
#Fitting the model as before
hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))


######################################################
########CONVOLUTIONAL NEURAL NETWORK:PLOT TRAINING#####
#######################################################
#Set the proportion
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
plt.savefig(CNN_dir + "/loss_acc_cross_entropy_normal.pdf")

################################################################################################################################################
##########################################################KULLBECK LIEBER-DIVERGENCE############################################################
################################################################################################################################################

#Set the seed: allow pseudo-randomness to be replicated
random.seed(34)

#Get list of unique labels, the index positions of when the first instance of the unique label is in the list[assumption: ordered list], count of unqiue labels 
unique,label_index_pos,label_counts = np.unique(label,return_index = True, return_counts = True)

#Delete the first position of the label_index_pos list as not used in spliting the label matrix
label_index_pos = np.delete(label_index_pos, 0) 

#Split the label matrix into arrays of unique label elements
split_label = np.split(label,label_index_pos)

#Get pmf for each image: if correct in thinking this is what is used for a label when using kullbeck lieber divergence
labels_prob = list()
labels_pmf = list()

for i in range(len(split_label)):
	x = unique
	xU, xL = x + 0.5, x - 0.5 
	prob = ss.norm.cdf(xU, loc = i, scale = 3) - ss.norm.cdf(xL, loc = i, scale = 3)
	prob = prob / prob.sum() #normalize the probabilities so their sum is 1
	labels_prob.append(list(prob))
	for j in range(len(split_label[i])):
		labels_pmf.append((prob))
	
x = unique 
fig = plt.figure()
for i in range(len(labels_prob)):
	values = labels_prob[i]
	plt.subplot(4,6,i + 1)
	width = 1/1.5
	plt.bar(x, values, width, color="blue")
	#plt.plot(x, values)
	plt.savefig(CNN_dir + "/label_kld.pdf")

#Shuffle the files, so that the labels still correspond
im_matrix,labels_pmf = sklearn.utils.shuffle(im_matrix,labels_pmf,random_state=2)

############################################################################
########################PROCESSING TRAINING DATA############################
############################################################################

################################################
######SPLIT TRAINING AND VALIDATION DATA########
################################################
#Determine the proportion of training data to be used for validation 
#Test_size: The proportion of files in the training im_matrix to use for training evaluation
val_size = float(data_split[1])/float(data_split[0]+data_split[1])
#Split the image matrices and the label pmf according to the specified proportions
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(im_matrix, labels_pmf, test_size=val_size, random_state=4)

################################################
###########PREPROSESSING THE  DATA##############
################################################
#Reshape the image matrices
#X_train.shape[0]: The number of samples used for training
X_train = X_train.reshape(X_train.shape[0], row_size, col_size, channels)
X_val = X_val.reshape(X_val.shape[0], row_size, col_size, channels)

#Rescale the data: result = data vectors lie in the range of [0,1]
#As images lie on the range of [0,255], we carry out simple rescaling by dividing the data by 255.
X_train /= 255
X_val /= 255

#Determine the number of classed used in analysis: the number of unique selection co-efficients
nb_classes = num_selection_coefficients

#pmf labels are in format of list of list: we need to convert to array of lists: DIMENSION
y_train=np.array([xi for xi in y_train])
y_val=np.array([xi for xi in y_val])

############################################################################
############################CNN ARCHITECTURE################################
############################################################################

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
#This model uses a linear sequential format
CNN = keras.models.Sequential()

#ROUND ONE OF PATTERN: CONV, MAX POOL, DROPOUT
#Declare the input layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last', input_shape=(row_size,col_size,channels)))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ROUND TWO OF PATTERN: CONV, MAX POOL, DROPOUT
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last'))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ROUND THREE OF PATTERN: CONV, MAX POOL, DROPOUT
#Add convolutional layer
CNN.add(keras.layers.convolutional.Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_last'))
#Add pooling layer: Max Pooling method (Parameter reduction)
CNN.add(keras.layers.convolutional.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))

#ADDING A FULLY CONNECTED LATER
#Weights of previous layers are flattened(made 1D) before passing to the fully connected dense layer
CNN.add(keras.layers.core.Flatten())
CNN.add(keras.layers.core.Dense(128, activation='relu'))
#Add dropout layer: regularises model to prevent over-fitting
CNN.add(keras.layers.core.Dropout(rate=0.5))
#Add output layer
CNN.add(keras.layers.core.Dense(nb_classes, activation='softmax'))
#Summary of the CNN architecture
CNN.summary()
#IMAGE OF THE MODEL
keras.utils.plot_model(CNN, to_file=CNN_dir +'/'+'model.png')


#######################################################
########CONVOLUTIONAL NEURAL NETWORK: COMPILE##########
#######################################################
#Compiling the neural network using KL divergence as a loss function. 
#Adams optimizer is still used
CNN.compile(loss='kld', optimizer='adam', metrics=['accuracy'])


#######################################################
########CONVOLUTIONAL NEURAL NETWORK:TRAINING##########
#######################################################
#Fitting the model as before
hist = CNN.fit(X_train, y_train, batch_size=batch_size_parameter, nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

######################################################
########CONVOLUTIONAL NEURAL NETWORK:PLOT TRAINING#####
#######################################################
#Set the proportion
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
plt.savefig(CNN_dir + "/loss_acc_kld.pdf")


#######################################################
######CONVOLUTIONAL NEURAL NETWORK: VALIDATION#########
#######################################################
#create list of files contained in each selection co-efficient folder
listing = listdir_nohidden(testing_dir)

#Count the number of images of each class: this takes into accoutnt that some classes fail to have entries in the testing data
training_data = list()
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
stop = NITER
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
np.save(testing_dir + '/' + "x_test",X_test_data,allow_pickle=False)
np.save(testing_dir + '/' + "y_test",y_test_data,allow_pickle=False)

#Evaluate the model using unseen testing data
score = CNN.evaluate(X_test_data ,y_test_data, batch_size = None, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



