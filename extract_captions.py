from argparse import ArgumentParser
import glob
import json
import os
import numpy as np
import torch
import re
import sys


def loadGloveModel():
    print("Loading Glove Model")
    gloveFile = 'glove/glove.6B.50d.txt'
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]], dtype=np.float32)
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_cap(caption):
    ''' Handles the word preprocessings. '''
    cap = caption.lower() # lowercase letters
    cap = cap.replace("'s", '') # remove possessive apostrophes ("woman's" -> "woman")
    cap = decontracted(cap)   # remove contractions "we're" -> "we are"
    cap = re.findall(r'[^\s!,.?@":;0-9()]+', cap)  # remove punctuation and symbols.
    return cap


def get_caption(ann_file, glove_model):
    ''' Returns the dictionary of video key values and the caption embeddings for each sentence. '''
    with open(ann_file, "r") as fobj:
        anet_v_1_0 = json.load(fobj)
    captions = {}
    not_found_list = set()
    for k, v in anet_v_1_0.items():
        caps = v["sentences"]
        sentences = []
        for cap in caps:
            cap = preprocess_cap(cap)
            cap_embed, not_found = get_glove(cap, glove_model)
            sentences.append(cap_embed)
            if len(not_found) > 0:
                not_found_list.update(not_found)
        captions[k] = sentences
    return captions, not_found_list


def get_glove(caption, glove_model):
    ''' Gets the caption string and returns the embedding of shape [CxL]
        C = Number of Glove features
        L = Number of words in the sentence
    '''

    C = 50  # We read the glove embeddings of 50 dim
    L = len(caption)
    embeddings = torch.Tensor(size=(C, L))
    j = torch.arange(C).long()
    not_found = []
    for c, w in enumerate(caption):
        if w in glove_model:
            embeddings[j, c] = torch.Tensor(glove_model[w])
        else:
            not_found.append(w)
    return embeddings, set(not_found)


if __name__ == "__main__":
    ann_file = "captions/train.json"
    glove_model = loadGloveModel()
    captions, not_found = get_caption(ann_file, glove_model)
    print("Number of unfound words: ", len(not_found), " : ", list(not_found)[:10])
    print("Number of captions", len(captions))