import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time

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

    return list(word_stems)




def main():
    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    print(raw_training_data)

    words = ["running", "runner", "runs", "happy", "happiness", "happenings", "banana", "hello"]
    print(preprocess_words(words, stemmer))


if __name__ == "__main__":
    main()
