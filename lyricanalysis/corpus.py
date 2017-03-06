# -*- coding: utf-8 -*-
"""corpus.py: Provides utility functions loading data from the billboard hot 100 list and lyrics."""

__author__      = "Tyler Marrs"

import os
import sys
import csv

from slugify import slugify


def root_dir():
    """Helper to get the root path of this project."""
    return os.path.join(os.path.dirname(__file__), os.pardir)


def data_dir():
    """Helper to get the data dir path of this project."""
    return os.path.join(root_dir(), 'data')


def _data_path_for_dir(name):
    """Helper to get data path given a dir string name."""
    return os.path.join(data_dir(), name)


def data_dirs(path):
    """Helper to get the list of directories in the given folder path."""
    data_dir = path
    dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))]
    
    return dirs


def swear_words():
    """Helper to get the list of swear words from the processed data dir.
    
    Returns a list of strings.
    """
    fp = os.path.join(data_dir(), 'swear-words.txt')
    words = []
    with open(fp, 'rt') as f:
        for line in f:
            words.append(line.strip())

    return words


def stop_words():
    """Helper to load custom list of stop words.
    
    Returns list of words.
    """
    fp = os.path.join(data_dir(), 'stopwords.txt')
    words = []
    with open(fp, 'rt') as f:
        for line in f:
            words.append(line.strip())
            
    return words


def songs_for_artist(artist, parent_dir=data_dir()):
    slugged = slugify(artist, only_ascii=True)    
    artist_dir = os.path.join(parent_dir, slugged)
    
    if not os.path.isdir(artist_dir):
        raise Exception(artist + " data does not exist.")
        
    return load_songs(artist_dir)


def load_songs(dir_path):
    """Loads a list of songs and their lyrics from text files into a dictionary given folder path.
    
    Returns list of dictionaries.
    """
    song_file = os.path.join(dir_path, 'songs.csv')
    if not os.path.isfile(song_file):
        song_file = os.path.join(dir_path, 'songs.tsv')
    
    first_row = True
    keys = []
    songs = []
    with open(song_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if first_row:
                first_row = False
                keys = row
                continue
                
            index = 0
            song = {}
            for index, value in enumerate(row):
                key = keys[index]
                song[key] = value
                
                if key == 'lyrics_file':
                    lyrics_file_path = os.path.join(dir_path, value)
                    with open(lyrics_file_path) as lf:
                        song['lyrics'] = lf.read()
                        song['lyrics_file_path'] = lyrics_file_path
            
            songs.append(song)
        
    return songs