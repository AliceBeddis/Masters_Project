#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Visualise the CNN: Kernel weights and activations

NOTE: THE INPUT X_training data loaded in the script must be of the same dimension as the data used to train the model.
"""

__author__ = 'Alice Beddis (alice.beddis14@imperial.ac.uk)'


import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas  as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from keras.models import Model
import operator
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.models import load_model
import numpy as np
import os
import subprocess
from tqdm import tqdm, trange
from time import sleep

################################################
matplotlib.rcParams.update({'figure.autolayout': True})
################################################
#Getting the terminal screen size: used when formatting user input
length_screen, width_screen = subprocess.check_output(['stty', 'size']).split()
width_screen  = int(width_screen)

##################################################################################################
##################################################################################################
##########################################FUNCTIONS###############################################
##################################################################################################
##################################################################################################


##########################################
#######FUNCTIONS FOR USER INPUT###########
##########################################

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

##########################################
###########CNN VISUALISATION##############
##########################################

def plot_weights(layer_index,no_filters,subplot_row, subplot_col):
	"""
	Plots the weights of filters of a given convolutional layer

	NOTE: THIS CODE WAS TAKEN FROM THE WEBPAGE: http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

	Keyword Arguments: 
		layer_index (int) -- the index of the layer being invesigated
		no_filters (int) -- the number of filters
		subplot_row (int) -- no. rows in subplot
		subplot_col (int) -- no. cols in subplot


	"""
	weight_conv2d_1 = model.layers[layer_index].get_weights()[0][:,:,0,:]
	layer_name = model.layers[layer_index].name
	#Get v_min and v_max for the colour bar
	ind_min = np.unravel_index(np.argmin(weight_conv2d_1, axis=None), weight_conv2d_1.shape)
	v_min = weight_conv2d_1[ind_min ]
	ind_max = np.unravel_index(np.argmax(weight_conv2d_1, axis=None), weight_conv2d_1.shape)
	v_max = weight_conv2d_1[ind_max]
	
	# Set up figure and image grid
	fig = plt.figure(figsize=(9.75, 3))
	grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
				 nrows_ncols=(subplot_row,subplot_col),
				 axes_pad=0.15,
				 share_all=True,
				 cbar_location="right",
				 cbar_mode="single",
				 cbar_size="7%",
				 cbar_pad=0.15,
				 )
	for i in range(no_filters):
		im = grid[i].imshow(weight_conv2d_1[:,:,i],cmap="gray", vmin =v_min, vmax= v_max )
	grid[i].cax.colorbar(im)
	grid[i].cax.toggle_label(True)
	plt.savefig(CNN_dir +'/'+ "Kernels" + str(layer_name) + ".pdf",bbox_inches='tight')


def display_activation(activations, col_size, row_size, act_index,no_filters): 
	activation = activations[act_index]
	layer_name = model.layers[act_index].name

	#Get v_min and v_max for the colour bar
	ind_min = np.unravel_index(np.argmin(activation, axis=None), activation.shape)
	v_min = activation[ind_min ]
	ind_max = np.unravel_index(np.argmax(activation, axis=None), activation.shape)
	v_max = activation[ind_max]

	# Set up figure and image grid
	fig = plt.figure(figsize=(row_size*2.5,col_size*1.5))
	grid = ImageGrid(fig, 111,
				 nrows_ncols=(row_size,col_size),
				 axes_pad=0.15,
				 share_all=True,
				 cbar_location="right",
				 cbar_mode="single",
				 cbar_size="7%",
				 cbar_pad=0.15,
				 )
	for i in range(no_filters):
		im = grid[i].imshow(activation[0, :, :, i], cmap=matplotlib.cm.binary, vmin =v_min, vmax= v_max )
	grid[i].cax.colorbar(im)
	grid[i].cax.toggle_label(True)
	plt.savefig(CNN_dir +'/'+ "Activation" + str(layer_name) + ".pdf",bbox_inches='tight')

def display_activation_col(activations, col_size, row_size, act_index,no_filters): 
	activation = activations[act_index]
	layer_name = model.layers[act_index].name

	#Get v_min and v_max for the colour bar
	ind_min = np.unravel_index(np.argmin(activation, axis=None), activation.shape)
	v_min = activation[ind_min ]
	ind_max = np.unravel_index(np.argmax(activation, axis=None), activation.shape)
	v_max = activation[ind_max]

	# Set up figure and image grid
	fig = plt.figure(figsize=(row_size*2.5,col_size*1.5))
	grid = ImageGrid(fig, 111,
				 nrows_ncols=(row_size,col_size),
				 axes_pad=0.15,
				 share_all=True,
				 cbar_location="right",
				 cbar_mode="single",
				 cbar_size="7%",
				 cbar_pad=0.15,
				 )
	for i in range(no_filters):
		im = grid[i].imshow(activation[0, :, :, i], cmap=matplotlib.cm.rainbow, vmin =v_min, vmax= v_max )
	grid[i].cax.colorbar(im)
	grid[i].cax.toggle_label(True)
	plt.savefig(CNN_dir +'/'+ "Activation_col_" + str(layer_name) + ".pdf",bbox_inches='tight')


##################################################################################################
##################################################################################################
#############################################MAIN#################################################
##################################################################################################
##################################################################################################

#GET THE RESULTS DIRECTORY

results_directory = getfile('Results Directory')
CNN_dir = results_directory + '/CNN'



############################################################################
#########################LOADING MODEL AND TRAINING DATA####################
############################################################################

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

#Get the number of filters in the model 
config = model.get_config()[0].get('config')
filters = config.get('filters')


#Load training data
X_train_data = np.load(CNN_dir +'/training_data/'+"X_train.npy")
#Choose input image + save it in the CNN directory
input_image = X_train_data[1:2,:,:,:]
fig = plt.figure()
plt.imshow(input_image[0,:,:,0],cmap='gray')
plt.savefig(CNN_dir +'/'+ "training_image.pdf",bbox_inches='tight')




#

############################################
##GETTING THE DIMENSIONS FOR THE SUBPLOTS###
############################################

#The number of subplots is determined by n_rows and n_height
print width_screen * "-" 
print('DETERMINING SUBPLOT DIMENSIONS'.center(width_screen)) 
print width_screen * "-" 

dimension_loop = True
print('{} subplots will be created on a row x col plot').format(filters)
while dimension_loop: 
	print width_screen * "-" 
	n_rows = inputNumber('n_rows')
	print width_screen * "-" 
	n_cols = inputNumber('n_cols')
	product_dimensions = n_rows*n_cols
	if product_dimensions >= filters:
		print width_screen * "-"     
		print "Choice: number of rows = {} | number of cols = {}".format(n_rows,n_cols)
		channels = 1
		dimension_loop = False
	else:
		print width_screen * "-" 
		print("The number of rows and columns are insufficient. Enter your choice again")



############################################################################
################################KERNEL WEIGHTS##############################
############################################################################
print width_screen * "-" 
print('PLOTTING KERNEL WEIGHTS'.center(width_screen)) 
print width_screen * "-"


for i in [0,3,6]:
	#Plot weights of first convo layer
	plot_weights(i,filters,n_rows,n_cols)
	#to prevent the script being to memory intensive we close the figure once it has been saved. 
	plt.clf()

# plot_weights(0,filters,n_rows,n_cols)	
# #to prevent the script being to memory intensive we close the figure once it has been saved. 
# plt.clf()
# #Plot weights of second convo layer
# plot_weights(3,filters,n_rows,n_cols)
# #to prevent the script being to memory intensive we close the figure once it has been saved. 
# plt.clf()
# #Plot weights of third convo layer
# plot_weights(6,filters,n_rows,n_cols)
# #to prevent the script being to memory intensive we close the figure once it has been saved. 
# plt.clf()

############################################################################
###############################ACTIVATION MAPS##############################
############################################################################
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(input_image)



print width_screen * "-" 
print('PLOTTING ACTIVATION MAPS: GREYSCALE'.center(width_screen)) 
print width_screen * "-"

for i in range(9):
	display_activation(activations,n_cols,n_rows,i,filters)
	#to prevent the script being to memory intensive we close the figure once it has been saved. 
	plt.clf()


print width_screen * "-" 
print('PLOTTING ACTIVATION MAPS: COLOUR'.center(width_screen)) 
print width_screen * "-"
for i in range(9):
	display_activation_col(activations,n_cols,n_rows,i,filters)
	#to prevent the script being to memory intensive we close the figure once it has been saved. 
	plt.clf()

