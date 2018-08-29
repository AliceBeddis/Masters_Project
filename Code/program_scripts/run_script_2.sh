#!/bin/bash
# This script runs python scripts with the aim of carrying out CNN analysis.


#############################################
#############GENERATING IMAGES###############
#############################################
read -p "Does this project involve training a CNN? (yes/no): " response
if [ $response == 'yes' ]
then
#############################################
########CHECKING IF INDEX FILE EXISTS########
#############################################
#Get the directory where simulations are stored: Ask until valid path is provided
read -p "Enter path to where simulations are stored: " path

while [ ! -d $path ]
do
echo "The path:'"$path"' does not exist:"
read -p "Enter path to where simulations are stored: " path
done

#Check to see if index file exists: FALSE(create_index.py executed), TRUE(check_sum.pu executed)
echo "Checking if index file exists"
if [ -e $path/data.json ]
then
echo "Updating Index File"
python check_sum_bash.py $path
else
echo "Creating Index File"
python create_index_bash.py $path
fi

#############################################
########GENERATING IMAGES FROM DATA #########
#############################################
echo "Generating Images"
#Get the path to the Results directory: Ask until valid path is provided
read -p "Enter path to results directory: " path_results

while [ ! -d $path_results ]
do
echo "The path:'"$path_results"' does not exist:"
read -p "Enter path to results directory: " path_results
done

python  create_images_options_bash.py $path $path_results

#############################################
########TRAINING CNN & PREDICTIONS###########
#############################################

echo "Training the CNN"
read -p "Enter name in which to save the trained model: " model

python basic_cnn_bash.py $path $path_results $model

echo "Make Predictions using CNN "
python predict_bash.py $path_results

else
read -p "Do images need to be generated from FASTA data? (yes/no): " response2
if [ $response2 == 'yes' ]
then
read -p "Enter path to where simulations are stored: " path

while [ ! -d $path ]
do
echo "The path:'"$path"' does not exist:"
read -p "Enter path to where simulations are stored: " path
done

read -p "Enter path to results directory: " path_results

while [ ! -d $path_results ]
do
echo "The path:'"$path_results"' does not exist:"
read -p "Enter path to results directory: " path_results
done


echo "Generating Images"
python  create_images_options_bash.py $path $path_results
fi
echo "Make Predictions using CNN "
python predict_bash.py $path_results

fi

exit









