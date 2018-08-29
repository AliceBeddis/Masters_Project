# Quantifying the Effect of Natural Selection on the Human Genome using Machine Learning

This project uses convolutional neural networks(CNNs) to quantify the effect of natural selection on genomes. The CNN is trained on data simulated using the MSMS coalescent simulation program. Real data in FASTA format can then be fed into the neural network to make quantitative predictions about the effect of natural selection. The confidence of predictions is quantified using Bayes Factors. 

## Getting Started


### Prerequisites

Python: version 2.7

Keras: version 2.1.3

Tensorflow: version 1.9.0

### Installing

Program assumes a basic Code, Data, Results directory strucutre. 

Code: Contains scripts for analysis: either the stand alone scripts or scripts for running the program(run_script_2.sh). 
Data: Contains MSMS or FASTA files. The index file(containing parameters used for all MSMS simulations) will also be stored in this directory.
Results:
        simulation_data: directory created when running scripts. Contains images created from simulation files. 
        real_data: directory created when running scripts. Contains images created from real data. 
        CNN: directory created when running scripts: Contains all files created when running CNN analysis. 
        
Running basic program( Code/program_scripts/run_script_2.sh):

Program is ran in the terminal.
The working directory must be the directory in which all program scripts are contained, eg. Code/program_scripts
Simply run the run_script_2.sh script.

## Authors


