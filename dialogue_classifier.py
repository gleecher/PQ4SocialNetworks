"""
Collaborators: Alex Tyson, Gannon Leech, Josh Lin
CSCI 3725 - Computational Creativity
Party Quest 4: Social Networks
Last Modified: 4/26/2021
Description: all the functions necessary to train/test a neural network for a dialogue classifier
Known Bugs: N/A
"""

# Imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time
from nltk.tokenize import RegexpTokenizer

def get_raw_training_data(filename):
    """Opens a CSV file and extracts its data into a dictionary structure. 
    This dictionary will have two keys: "person", which is paired with the
    name of the actor who is speaking, and "sentence", which is paired 
    with a sentence that person has said.
    
    All entries are lowercase to make actor word comparison easier.

    Args:
        filename (str): a csv file containing data
    
    returns training_data: a list of dictionaries 
    """
    training_data = []
    filename = "inputs/" + filename
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            line_info = {'person': row[0].lower(), 'sentence': row[1].lower()}
            training_data.append(line_info)
    
    return training_data


def preprocess_words(words, stemmer):
    """Given a list of words, stems each word and returns a list 
    of the word stems without duplicates."""
    
    # Note: unwanted tokens?
    word_stems = set()

    for word in words:
        stem = stemmer.stem(word)
        word_stems.add(stem)

    return sorted(list(word_stems))

def organize_raw_training_data(raw_training_data, stemmer):
    """For each element of our training_data list... 
    
    1) Retrieve list of tokens from the sentence
    2) Add list of tokens to "words" list
    3) Add (token_list, actor_name) tuple to list of "documents" list
    4) Add never-before-seen actors to "classes" list (no duplicates!)

    Afterwards, finds the stems of all words in the "words" list

    Returns "words", "documents", and "classes" lists
    """
    stem_list = []
    word_list = []
    actor_list = []
    documents = []
    
    tokenizer = RegexpTokenizer(r'\w+')
    for line_dict in raw_training_data:
        actor = line_dict['person']
        sentence = line_dict['sentence']
        
        if actor not in actor_list:
            actor_list.append(actor)
        
        token_sent = tokenizer.tokenize(sentence)
        for word in token_sent:
            word_list.append(word)
  
        documents.append((token_sent, actor))
    
    # Modify our list of words to contain word stems
    stem_list = preprocess_words(word_list, stemmer)
    
    return stem_list, actor_list, documents


def create_training_data(word_stems, classes, documents, stemmer):
    """Given some organized raw training data, returns a list of
    training data formatted according to the Bag of Words model. 

    Output matches a sentence to the actor who said it 
    (as provided with the classes parameter)
    """
    output = []
    training_data = []
    
    for pair in documents:
        sentence_stems = preprocess_words(pair[0], stemmer)
        bag = []

        for word in word_stems:
            if word in sentence_stems:
                bag.append(int(1))
            else:
                bag.append(int(0))
        
        actors = []
        for actor in classes:
            if actor == pair[1]:
                actors.append(int(1))
            else:
                actors.append(int(0))
    
        output.append(actors)
        training_data.append(bag)
        
    return training_data, output


def sigmoid(z):
    """Calculate the sigmoid for a given input z"""
    return 1 / (1 + np.exp(-z))


def sigmoid_output_to_derivative(output):
    """Convert the sigmoid function's output to its derivative.
    
    This function was provided in the lab handout"""
    return output * (1-output)


def main():
    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('dialogue_data.csv')

    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    print("words: " + str(words))
    print("classes: " + str(classes))
    print("documents: " + str(documents))

    training_data, output = create_training_data(words, classes, documents, stemmer)
    print("Training data: " + str(training_data))
    print("\n\nOUTPUT: " + str(output))

if __name__ == "__main__":
    main()
