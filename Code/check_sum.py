#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
Performs check-sum operation to determine if files have been modified added or deleted and updates the index file
"""

__author__ = 'Alice Beddis'

import os
import json 
import platform
import gzip 
from operator import itemgetter

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
	order = 'Type the path to the %s: ' %(filetype)
	error = 'The path to the %s entered does not exist' %(filetype)
	user_input = raw_input(order)
	user_input = user_input.strip()
	result = os.path.exists(user_input)
	if result == False: 
		print error
		getfile('%s'%(filetype))  #Repeat the function recursively
	else: 
		return user_input

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

################################################
############COLLECTING THE META_DATA############
################################################

#Get the name of the directory containing the simulation folders and the index_json file
simulation_folder = getfile('folder containing simulation files')

#Geting dictionary of current simulation files and thier associated time-stamps
simulation_check_sum = []

for filename in os.listdir(simulation_folder): 
	#Ignore operating system files( prevents .DSStore files from being read/ might not need now that I have the two if statements below)
	if not filename.startswith('.'):
		if not filename.endswith('.json'): 
			string = simulation_folder + '/%s' %(filename) #
			#Get the name of each of the simulation files 
			name = filename
			#Get the UNIX modification time stamp for each of the simulation files 
			modication_stamp = modification_date(string)
			#Create Dictionary of simulation specific paramters
			dictionary_temp = createDict('name','modication_stamp')
			#For each simulation file, append its specific dictionary to a list
			simulation_check_sum.append(dictionary_temp)

#Open the index_file
with open(simulation_folder +'/data.json', 'r') as fp:
	data = json.load(fp)

#######################################################
##########DETECT INTERSECTION AND DIFFERENCES##########
#######################################################

#List of file names in index file
old = [d['name'] for d in data if 'name' in d]
#List of file names currently in directory containing the simulation files
new = [d['name'] for d in simulation_check_sum if 'name' in d]
#List of new simulation files
new_files = list(set(new) - set(old))
#List of new simulation files removed from the directory since the index file was created
removed_files = list(set(old) - set(new))
#Files that remian in the 
shared_files = list(set(old).intersection(new))

################################################
##########UPDATING INDEX: NEW FILES#############
################################################

#Suffix of g_zip files (Compressed)
gzip_suffix = "gz" 
#Suffix of txt files (Uncompressed)
txt_suffix = "txt"
suffixes = (gzip_suffix, txt_suffix )
new_files = [x for x in new_files if x.startswith(suffixes)]
for new_file in new_files:
	#Path to each of the new files
	path_new_file = simulation_folder + '/' + new_file
	#Open the files in appropriate way and read the first line
	if new_file.endswith(gzip_suffix):
		with gzip.open(path_new_file, 'rb') as f:
			first_line = f.readline()
	elif new_file.endswith(txt_suffix):
		with open(path_new_file) as f:
			first_line = f.readline()
	#Extract the parameter values
	NREF = substring_after(first_line, '-N', 0)
	NCHROM = substring_after(first_line, '-N', 1)
	NITER = substring_after(first_line, '-N', 2)
	SAA = substring_after(first_line, '-SAA',0)
	SAa = substring_after(first_line, '-SAa',0)
	Saa = substring_after(first_line, '-Saa',0)
	P_Recombination = substring_after(first_line, '-r',0)
	no_recom_sites = substring_after(first_line, '-r',1)
	t = substring_after(first_line, '-t',0)
	name = new_file
	active = 'active' 
	modication_stamp = modification_date(path_new_file)
	#Create Dictionary of simulation specific paramters
	dictionary_temp = createDict('name','no_recom_sites','P_Recombination', 'NREF', 'NCHROM','NITER','SAA','SAa','Saa', 't','modication_stamp','active')
	#Add the dictionary of new file parameter values to the index file
	data.append(dictionary_temp)


################################################
########UPDATING INDEX: REMOVED FILES###########
################################################

#For each of the removed files, set them to inactive in the index file: allows metadata to be recorded
active = 'inactive'

removed_files = [x for x in removed_files if x.startswith(suffixes)]

for removed_file in removed_files: 
	removed_dict = filter(lambda x: x.get('name') == removed_file, data)
	removed_dict[0]['active'] = 'inactive'
	#Find the index of the file dictionary
	removed_index = next((index for (index, d) in enumerate(data) if d["name"] == removed_file), None)
	#Update the files meta_data
	data[removed_index] = removed_dict[0]

################################################
########UPDATING INDEX: MODIFIED FILES##########
################################################

#Create two lists, one with file names and the current modification time stamps and one with the index file time stamps
old_modification = [] 
new_modification = []

shared_files = [x for x in shared_files if x.startswith(suffixes)]
for shared in shared_files: 
	common_dict_old = filter(lambda x: x.get('name') == shared, data)
	common_dict_new = filter(lambda x: x.get('name') == shared, simulation_check_sum )	
	old_modification.append({k: common_dict_old[0].get(k, None) for k in ('name','modication_stamp')})
	new_modification.append(common_dict_new[0])

#Sort to ensure that the order of the two lists is exactly the same: an assumption of the following operation
old_modification, new_modification = [sorted(l, key=itemgetter('name')) for l in (old_modification, new_modification)]

#Compare the two lists: create list of file names that have been modified
pairs = zip(old_modification, new_modification )
if any(x != y for x, y in pairs) == True:
	modified_file_names = [y['name'] for x, y in pairs if x['modication_stamp'] != y['modication_stamp']]

#For each modified file, create a new dictionrary entry and replace the old out of date files index entry
	for modified_entry in modified_file_names: 
		modified_path = simulation_folder + '/' + modified_entry
		if modified_entry.endswith(gzip_suffix):
			with gzip.open(modified_path, 'rb') as f:
				first_line = f.readline()
		elif modified_entry.endswith(txt_suffix):
			with open(modified_path) as f:
				first_line = f.readline()

		NREF = substring_after(first_line, '-N', 0)
		NCHROM = substring_after(first_line, '-N', 1)
		NITER = substring_after(first_line, '-N', 2)
		SAA = substring_after(first_line, '-SAA',0)
		SAa = substring_after(first_line, '-SAa',0)
		Saa = substring_after(first_line, '-Saa',0)
		P_Recombination = substring_after(first_line, '-r',0)
		no_recom_sites = substring_after(first_line, '-r',1)
		t = substring_after(first_line, '-t',0)
		name = modified_entry
		active = 'active' # This allows previously deleted files that perhaps have been re-created to be updated
		modication_stamp = modification_date(simulation_folder + '/' + modified_entry)
		#Create Dictionary of simulation specific paramters
		dictionary_temp = createDict('name', 'NREF', 'NITER','NCHROM','P_Recombination','no_recom_sites','Saa','SAA','SAa', 't','modication_stamp','active')

		modfied_index = next((index for (index, d) in enumerate(data) if d["name"] == modified_entry), None)

		data[modfied_index] = dictionary_temp


with open(simulation_folder + '/data.json', 'w') as fp:
	json.dump(data, fp, sort_keys=True, indent=4)

