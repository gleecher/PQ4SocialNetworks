# PQ4SocialNetworks
Using neural networks to classify movie quotes from specific actors!

## Collaborators: 
Alex Tyson, Gannon Leech, Josh Lin

CSCI 3725 - Computational Creativity // Party Quest 4: Social Networks

Last Modified: 5/1/21

## Title: 
Actor Dialogue Classifier

## Running the Code: 
After cloning this repository to your local machine...

To set up the code with your desired inputs: <br>
1) Upload your input file into the *inputs* folder

2) In the main function of the main.py file, you can add the name of the desired input file by entering it as the argument of the raw_training_data method. (By default, we use dialogue_data.csv) 

Then later on in the main method, you can add additional lines for the program to classify by calling the classify() method. Here, the line you would like to classify is the third function argument. 

In order to run the code you must run the main.py file. The program will execute and print the its classifications for each line that was requested to be classified. 

In order to speed up code run time, if you have run the start_training method once already on your input file, you can comment it out when classifying additional phrases as the file saves the synapses for its last used training run.


## General Description:  
This program uses a neural network to classify and identify the speakers of given phrases. Using a given input file, the program classifies every phrase by who the speaker is, and then classifies the sentence using a bag of words representation. Additionally, it preprocesses the information by using the stems of every word instead of the words themselves. 

The main file then uses all of that information to train a neural network, with the goal of being able to classify different phrases based on the training set.



