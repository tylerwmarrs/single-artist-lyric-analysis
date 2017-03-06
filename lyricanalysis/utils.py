import unicodedata
import sys

from nltk.stem import PorterStemmer

def split_sentences(text):
    sentences = []
    for sentence in text.split('\n'):
        sentence = sentence.strip()

        if sentence:
            sentences.append(sentence)

    return sentences
    
    
stemmer = PorterStemmer()    
def stem_words(words):
    return [stemmer.stem(w) for w in words]


punc_tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(punc_tbl)


def remove_stop_words(stop_words, words):
    """Remove stop words from a list of words."""
    wl = []    
    for word in words:
        word = word.lower()
        if word not in stop_words:
            wl.append(word)

    return wl

def song_repetiveness(lyrics, rate=2):
    # split song on sentence and find unique sentences
    sentences = split_sentences(lyrics)
    unique_sentences = set(sentences)

    total_sentences = len(sentences)
    total_unique_sentences = len(unique_sentences)

    # collect frequency of unique sentences and calculate reptetiveness
    repetitive_rate = 0
    frequency = 0
    for usentence in unique_sentences:
        for sentence in sentences:
            if usentence == sentence:
                frequency = frequency + 1

        # only calc. reptetiveness rate if frequency rate cutoff is met
        if frequency >= rate:
            repetitive_rate = repetitive_rate + (frequency / total_sentences)

        frequency = 0

    return repetitive_rate