#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Takes MSMS simulation txt files and extracts paramter values: stores in json format.
"""

__author__ = 'Alice Beddis'

import os
import json 
import platform
import gzip 

##################################################################################################
##################################################################################################
##########################################FUNCTIONS###############################################
##################################################################################################
##################################################################################################

def createDict(*args):
	"""
	Creates dictionary from varaibles: Key = Variable name, Value = Variable contents
	
	Keyword Arguments: 
		args(string/int/float) -- varaible to convert into dictionary format
	
	Returns: 
		dict(dictionary) -- dictionary of variables

	"""
	return dict(((k, eval(k)) for k in args))

def getfile(filetype):
	"""
	Function to check if the input file/folder exists: if not asks for new file path until correct one found
	
	Keyword Arguments: 
		filetype (string) -- type of file looking for, eg 'reference genome'
	
	Returns: 
		user_input (string) -- path to specifed files

	"""
	order = 'Type the path to the %s :' %(filetype)
	error = 'The path to the %s entered does not exist' %(filetype)
	user_input = raw_input(order)
	user_input = user_input.strip()
	result = os.path.exists(user_input)
	if result == False: 
		print error
		getfile('%s'%(filetype))  #Repeat the function recursively
	else: 
		return user_input

#def substring_after(string, flag):
#	"""
#	Function to extracts the parameter value IMMEDIATELY AFTER the specified parameter flag
#
#	Keyword Arguments:
#		string(string) -- string in which data is to be extracted from
#		flag(string) -- string specifying the paramter flag
#
#	Returns:
#		extracted_parameter(string) -- Extracted parameter value
#
#	"""
#	sub_string = string.partition(flag)[2]
#	sub_string_list = sub_string.split()
#	extracted_parameter = sub_string_list[0]
#	return extracted_parameter;

def substring_after(string, flag, positions):
	"""
	Function to extracts the parameter value X positions after the specified parameter flag
	
	Keyword Arguments: 
		string(string) -- string in which data is to be extracted from
		flag(string) -- string specifying the paramter flag
		positions(int) -- number of postions after the flag to extract paramter value: 0 = immediately next, 1 = 2 postions after....
	
	Returns: 
		extracted_parameter(string) -- Extracted parameter value

	"""
	sub_string = string.partition(flag)[2]
	sub_string_list = sub_string.split() 
	extracted_parameter = sub_string_list[positions]
	return extracted_parameter;


def modification_date(path_to_file):
	"""
	Get the UNIX Time Stamp of when the file was modification
	"""
	if platform.system() == 'Windows':
		return os.path.getmtime(path_to_file)
	else:
		stat = os.stat(path_to_file)
		return stat.st_mtime

##################################################################################################
##################################################################################################
############################################MAIN##################################################
##################################################################################################
##################################################################################################
#List stores the parameter value dictionaries
simulations = [] 
#Suffix of g_zip files (Compressed)
gzip_suffix = "gz" 
#Suffix of txt files (Uncompressed)
txt_suffix = "txt" 

#Get the path to the directory containing the simuatlion files
simulation_folder = getfile('folder containing simulation files')

#Open the directory in which simulation files are stored
for filename in os.listdir(simulation_folder): 
	#Ignore operating system files( prevents .DSStore files from being read/ might not need now that I have the two if statements below)
	if not filename.startswith('.'): 
		string = simulation_folder + '/%s' %(filename) #
		#Open in appropriate way, depending if g_zipped or uncompressed
		if string.endswith(gzip_suffix):
			with gzip.open(string, 'rb') as f:
				first_line = f.readline()
		elif string.endswith(txt_suffix):
			with open(string) as f:
				first_line = f.readline()
		else:
			continue
			
		#Extracting parameters	
		NREF = substring_after(first_line, '-N', 0)
		NCHROM = substring_after(first_line, '-N', 1)
		NITER = substring_after(first_line, '-N', 2)
		SAA = substring_after(first_line, '-SAA',0)
		SAa = substring_after(first_line, '-SAa',0)
		P_Recombination = substring_after(first_line, '-r',0)
		no_recom_sites = substring_after(first_line, '-r',1)
		t = substring_after(first_line, '-t',0)
		name = filename
		modication_stamp = modification_date(string)
		active = 'active' # This allows deleted files to be tracked in json folder
		#Create Dictionary of simulation specific paramters
		dictionary_temp = createDict('name','no_recom_sites','P_Recombination', 'NREF', 'NCHROM','NITER','SAA','SAa', 't','modication_stamp','active')
		#For each simulation file, append its specific dictionary to a list
		simulations.append(dictionary_temp)

#Save the simulation meta-data in json format
with open(simulation_folder + '/data.json', 'w') as fp:
	json.dump(simulations, fp, sort_keys=True, indent=4)









