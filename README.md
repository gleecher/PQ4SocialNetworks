# PQ4SocialNetworks

Collaborators: Alex Tyson, Gannon Leech, Josh Lin

CSCI 3725 - Computational Creativity

Party Quest 4: Social Networks

Last Modified: 4/26/21

Title:

Running the Code:
To set up the code with your desired inputs, in the main function of the main.py file, you can add the desired input file you would like by entering it as the second parameter of the raw_training_data method. Then later on in the main method, you can add additional lines for the program to classify by adding more classify methods, where the line you would like to classify is the third parameter. In order to run the code you must run the main.py file. The program will execute and print the its classifications for each line that was requested to be classified. In order to speed up code run time, if you have run the start_training method once already on your input file, you can comment it out when classifying additional phrases as the file saves the synapses for its last used training run.


General Description: 
This program uses a neural network to classify and identify the speakers of given phrases. Using a given input file, the program classifies every phrase by who the speaker is, and then classifies the sentence using a bag of words representation. Additionally, it preprocesses the information by using the stems of every word instead of the words themselves. 

The main file then uses all of that information to train a neural network, with the goal of being able to classify different phrases based on the training set.



