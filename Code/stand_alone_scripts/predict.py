#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Makes Predictions using trained CNN

#python version 2.17
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import os.path
import glob
import tensorflow
import keras#Note: Thhe version number has to be 2.1.3
from keras import backend as K
from keras.models import load_model
import pandas as pd
import subprocess
import tqdm
import time
import subprocess
import PIL 
import numpy as np
import json
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


##################################################################################################
##################################################################################################
##########################################FUNCTIONS###############################################
##################################################################################################
##################################################################################################

##########################################
####FUNCTIONS FOR PROCESSING FILES########
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


##################################################################################################
##################################################################################################
#############################################MAIN#################################################
##################################################################################################
##################################################################################################

############################################################################
################################GET DIRECTORY STRUCTURE#####################
############################################################################


##########################################
################GET FILE PATHS############
##########################################
print width_screen * "-" 
print('FILE PATHS'.center(width_screen)) 
print width_screen * "-" 
#Get the location of the simulations 
simulation_files = getfile('Simulation Files')
print width_screen * "-" 
#Get the location of the results directory
results_directory = getfile('Results Directory')
print width_screen * "-" 
#Get the location of the directory containing real data images
real_data_input = getfile('real_data_images')
print width_screen * "-" 
#Using the path to the results directory we get the path to the simulation images
simulation_images = results_directory + '/simulation_data'
print width_screen * "-" 
#Using the path to the results directory we get the path to the CNN directory
CNN_dir = results_directory + '/CNN'

############################################################################
################################LOAD MODEL##################################
############################################################################
print width_screen * "-" 
print('LOADING MODEL'.center(width_screen)) 
print width_screen * "-" 

#Get the location of the saved model
model_path = getfile('CNN Model')
print width_screen * "-" 

#Load the specified model
model = load_model(model_path)


#If exists: load the label_selection co-efficient association file
base_name = os.path.basename(os.path.basename(model_path))
(base_root,ext) = os.path.splitext(base_name)
#
label_association_path = CNN_dir + '/' + base_root + '_class_label_association.json'

	
#Determine the row_size and col_size used in training of the model 
config = model.get_config()[0].get('config')
#Get the batch input shape: contains [rows, cols, channels]
batch_shape = list(config.get('batch_input_shape'))
#Get the row size 
img_rows = batch_shape[1]
#Get the col size 
img_columns = batch_shape[2]
#Get the channels 
channels = batch_shape[3]

############################################################################
################################LOAD DATA###################################
############################################################################
list_real_data_images = listdir_nohidden(real_data_input)

#Process the images and normalise them 
real_data = []
for image_file in list_real_data_images:
	im = PIL.Image.open(image_file)
	im_resized = im.resize((img_columns,img_rows))
	im_array = np.asarray(im_resized).flatten()
	im_test = im_array.reshape(1, img_rows, img_columns,channels)
	im_test = im_test.astype('float32')
	im_test /= 255
	real_data.append(im_test)

############################################################################
################################PREDICTION##################################
############################################################################

#Method1: just get the class predictions 
class_predictions = map(model.predict_classes, real_data)
class_predictions  = [item for sublist in class_predictions  for item in sublist]
#Only get the last part of the file path 
image_name = map(os.path.normpath, list_real_data_images)
image_name_2 = map(os.path.basename, image_name)
#Create dictionary of image name and class prediction
prediction_dictionary = dict(zip(image_name_2, class_predictions))

#Method2: Get the probability of each data instance belonging to each class
prob_predictions = map(model.predict_proba, real_data)
prob_predictions_flat  = [item for sublist in prob_predictions  for item in sublist]
#We use the list of image names used above to create a dictionary
prob_prediction_dictionary = dict(zip(image_name_2, prob_predictions_flat))



if os.path.isfile(label_association_path):
	with open(label_association_path, 'r') as fp:
			label_association_dict= json.load(fp)
	true_class_predictions = list()
	#Replace the label with selection co-efficients in the prediction dictionary
	for label in class_predictions:
		sel_cof = label_association_dict.get(str(label))
		true_class_predictions.append(sel_cof)
	prediction_dictionary = dict(zip(image_name_2, true_class_predictions))
	int_labels = map(int, label_association_dict.keys())
	int_labels.sort()

split_names = [x[0] for x in map(os.path.splitext, image_name_2)]

############################################################################
##############################BAYES FACTOR##################################
############################################################################
#Bayes factor is a statistical index that qunatified the level of support for a hypothesis
#When the two hypotheses are equally probable a priori the BF is equal to the ratio of the posterior probabilities of the hypotheses 



#####################################################
#######BAYES FACTOR: HYPOTHESIS = NO SELECTION#######
##################################################
print width_screen * "-" 
print('BAYES FACTOR CALCULATION'.center(width_screen)) 
print width_screen * "-" 
##Hypothesis 1 : No Selection 
##Hypothesis 2 : Selection 
print width_screen * "-" 
print('BAYES FACTOR: HYPOTHESIS = NO SELECTION'.center(width_screen)) 
print width_screen * "-" 

M_1_list = list()
M_2_list = list()
K_list = list()
statement_list = list()

for prediction in range(len(prob_predictions_flat)):
	M_1 = prob_predictions_flat[prediction][0]
	M_1_list.append(M_1)
	M_2 = sum(prob_predictions_flat[prediction][1:])
	M_2_list.append(M_2)
	K = M_1/M_2
	K_list.append(K)
	if K <0: 
		statement =  'Negative Support for No Selection'
	elif K >= 0 and K <= 5:
		statement =  'Minimal Support for No Selection'
	elif K  > 5 and K <= 10:
		statement =  'Substantial Support for No Selection'
	elif K  > 10 and K <= 15:
		statement =  'Strong Support for No Selection'
	elif K  > 15 and K <= 20:
		statement =  'Very Strong Support for No Selection'
	elif K  > 15 and K <= 20:
		statement =  'Very Strong Support for No Selection'
	elif K  > 20:
		statement =  'Decisive Support for No Selection'
	statement_list.append(statement)

Bayes_factor_no_selection= pd.DataFrame(
    {'Population': split_names,
     'M1': M_1_list,
     'M2': M_2_list,
     'Bayes Factor': K_list,
     'Support' : statement_list
    })

print(Bayes_factor_no_selection.to_string().center(width_screen))
Bayes_factor_no_selection.to_csv(CNN_dir + '/Bayes_factor_no_selection.csv',index=False)


#####################################################
#######BAYES FACTOR: HYPOTHESIS = CLASS PREDICTION###
#####################################################

print width_screen * "-" 
print('BAYES FACTOR: HYPOTHESIS = CLASS PREDICTION'.center(width_screen)) 
print width_screen * "-" 

M_1_list = list()
M_2_list = list()
K_list = list()
statement_list = list()

for prediction in range(len(prob_predictions_flat)):
	M_1 = prob_predictions_flat[prediction][class_predictions[prediction]]
	M_1_list.append(M_1)
	M_2 = sum(np.delete(prob_predictions_flat[prediction], class_predictions[prediction]))
	M_2_list.append(M_2)
	K = M_1/M_2
	K_list.append(K)
	if K <0: 
		statement =  'Negative Support for Selection Coefficient of {}'.format(true_class_predictions[prediction])
	elif K >= 0 and K <= 5:
		statement =  'Minimal Support for Selection Coefficient of {}'.format(true_class_predictions[prediction])
	elif K  > 5 and K <= 10:
		statement =  'Substantial Support for Selection Coefficient of {}'.format(true_class_predictions[prediction])
	elif K  > 10 and K <= 15:
		statement =  'Strong Support for No Selection Coefficient of {}'.format(true_class_predictions[prediction])
	elif K  > 15 and K <= 20:
		statement =  'Very Strong Support for Selection Coefficient of {}'.format(true_class_predictions[prediction])
	elif K  > 15 and K <= 20:
		statement =  'Very Strong Support for Selection Coefficient of {}'.format(true_class_predictions[prediction])
	elif K  > 20:
		statement =  'Decisive Support for Selection Coefficient of {}'.format(true_class_predictions[prediction])
	statement_list.append(statement)

Bayes_factor_predicted_class= pd.DataFrame(
    {'Population': split_names,
     'Predicted Class': true_class_predictions,
     'Predicted Label': class_predictions,
     'M1': M_1_list,
     'M2': M_2_list,
     'Bayes Factor': K_list,
     'Support' : statement_list
    })

print(Bayes_factor_predicted_class.to_string().center(width_screen))
Bayes_factor_predicted_class.to_csv(CNN_dir + '/Bayes_factor_predicted_class.csv',index=False)
