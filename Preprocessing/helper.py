import string
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
nlp = spacy.load('en_core_web_md')

import wordninja
import pickle as pkl
import re
import json
import contractions

from tqdm import tqdm

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import 	WordNetLemmatizer


word_dict = pkl.load(open('Data/emnlp_dict.pkl', 'rb'))
wordnet_lemmatizer = WordNetLemmatizer()

# setting up the stopwords list
negation_words = [
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
    "hardly", "scarcely", "barely", "rarely", "seldom", "little", "don't", "can't",
    "won't", "isn't", "aren't", "couldn't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "shouldn't", "wouldn't", "mustn't", "mightn't"
]
non_stop_words = ['user', 'url', '<user>', '<url>']
stop_words = list((set(stopwords.words('english')) - set(negation_words)) - set(non_stop_words))

# setting up the emoticon dictionary
emoticon_mapping = {
    r':-?\)': ' <happy> ',
    r':-?\(': ' sad ',
    r':-?D': ' big grin ',
    r':-?P': ' <tongue out> ',
    r';-?\)': ' <wink> ',
    r':\'?\(': ' <crying> ',
    r':-\|': ' <neutral> ',
    r':-\/': ' <skepticism or disapproval> ',
    r':\'\)': ' tears of joy ',
    r':-O': ' <surprise> ',
    r':-\*': ' <kiss> ',
    r'8-\)': ' <cool or sunglasses> ',
    r'<3': ' <love> ',
    r'\^-^': ' <happy or excited> ',
    r'XD': ' <laughing> ',
    r'-_-\'': ' <disapproval or disbelief> ',
    r':v': ' <pac-man> ',
    r'¯\\_(ツ)_/¯': ' <shrug or meh> ',
    r'\.\.\.': ' pause ',
    
    # Variations with space between face and mouth
    r': -?\)': '<happy>',
    r': -?\(': '<sad>',
    r': -?D': '<big grin>',
    r': -?P': '<tongue out>',
    r'; -?\)': '<wink>',
    r': \'?\(': '<crying>',
    r': -\|': '<neutral>',
    r': -\/': '<skepticism or disapproval>',
    r': -O': '<surprise>',
    r': -\*': '<kiss>',
    r'8 -?\)': '<cool or sunglasses>',
    r':\' \)': ' tears of joy ',
    # regex that matches anything between < and >
    # r'<[^>]*>': ''
}

# setting up the slang dictionary
with open('Preprocessing/ShortendText.json', 'r') as f:
    slang_dict = json.load(f)


def apply_spacy(texts: list)->list:
    """
    Performs a series of checks to correct a word
        - first checks if the word is in the dictionary
        - segments the word
        - translates the segments from slang to english
        - corrects spelling mistakes
        - if word still not in dictionary, take the first suggestion given by enchant
    :param text: text to correct
    :return: corrected text
    """
    result = []
    for doc in tqdm(nlp.pipe(texts)):
        result.append(' '.join([token.text for token in doc]))

    return result

def decontracted(text: str) -> str:
    return contractions.fix(text)


def correct_text(text: list)->list:
    text = [slang_dict.get(word, word) for word in text]
    text = [word_dict.get(word, word) for word in text]
    # text = [TextBlob(word).correct().raw for word in text]
    return text

def replace_link(text: str) -> str:
    """
    replaces all links in a string with the word link
    :param text: text to replace links in
    :return: text with replaced links
    """
    return re.sub(r'http\S+', ' link ', text)


def remove_punctuation(text: str) -> str:
    """
    removes all punctuation from a string
    :param text: text to remove punctuation from
    :return: text without punctuation
    """
    # replace all punctiation except ' with a space
    text = text.translate(str.maketrans(string.punctuation.replace('\'', ''), ' ' * (len(string.punctuation) - 1)))
    # remove double spaces
    text = re.sub(' +', ' ', text)
    return text

def replace_hashtag(text: str)-> str:
    """
    Splits the text by space and check if word starts with # and if it does it replaces it with talking about
    and segment the word right after the #
    """
    words = text.split()
    for i in range(len(words)):
        if words[i].startswith('#'):
            words[i] = slang_dict.get(words[i], words[i])
            words[i] = word_dict.get(words[i], words[i])
            words[i] = ' hashtag ' + ' '.join(wordninja.split(''.join(correct_word([words[i][1:]]))))
    return ' '.join(words)

def list_to_string(text: list) -> str:
    """
    converts a list to a string
    :param text: list to convert
    :return: string
    """
    return ' '.join(text)

def string_to_list(text: str) -> list:
    """
    converts a string to a list
    :param text: string to convert
    :return: list
    """
    return text.split()

def identity(text: str) -> str:
    """
    returns the text as it is
    :param text: text to return
    :return: text
    """
    return text

def slang_to_english(text: list) -> list:
    """
    converts slang words to english
    :param text: text to convert
    :return: converted text
    """
    # open the slang json

    # replace slang words with english
    return [slang_dict.get(word, word) for word in text]

def replace_emoticons(text: str) -> str:
    """
    replaces emoticons with their meaning
    :param text: text to replace emoticons in
    :return: text with replaced emoticons
    """
    for emoticon, meaning in emoticon_mapping.items():
        text = re.sub(emoticon, meaning, text)
    return text

def replace_unmatched_parentheses(text:str) -> str:
    stack = []
    result = ""
    
    for char in text:
        if char == '(':
            stack.append(char)
            result += char
        elif char == ')':
            if len(stack) > 0:
                stack.pop()
                result += char
            else:
                result += ' smiling '
        else:
            result += char
    
    while len(stack) > 0:
        result += ' parenthesis '
        stack.pop()
    
    return result

def remove_stopwords(text: list) -> list:
    """
    removes all stopwords from a string
    :param text: text to remove stopwords from
    :return: text without stopwords
    """
    return [word for word in text if word not in stop_words]

def to_lowercase(text: str) -> str:
    """
    converts a string to lowercase
    :param text: text to convert to lowercase
    :return: text in lowercase
    """
    return text.lower()

def make_two_consecutive(text : str) -> str:
    modified_string = re.sub(r'([bcdefghjklmnopqrstvwxyz])\1{2,}', r'\1\1', text)
    modified_string = re.sub(r'([aiu])\1+', r'\1', modified_string)
    return modified_string

def segment_words(text: string) -> list:
    """
    segments words in a list
    :param text: text to segment
    :return: segmented text
    """
    return wordninja.split(text)

def correct_word(word: list)->list:
    return [word_dict.get(w, w) for w in word]

# def replace_hashtag(text: str)-> str:
#     return re.sub(r'#', ' talking about ', text)


def remove_numbers(text: str) -> str:
    """
    removes all numbers from a string
    :param text: text to remove numbers from
    :return: text without numbers
    """
    return ''.join([i for i in text if not i.isdigit()])

def remove_whitespace(text: str) -> str:
    """
    removes all whitespaces from a string
    :param text: text to remove whitespaces from
    :return: text without whitespaces
    """
    return ' '.join(text.split())

def remove_single_characters(text: str) -> str:
    """
    removes all single characters from a string
    :param text: text to remove single characters from
    :return: text without single characters
    """
    return ' '.join([word for word in text.split() if len(word) > 1])

def remove_short_words(text: str) -> str:
    """
    removes all words with less than 3 characters from a string
    :param text: text to remove short words from
    :return: text without short words
    """
    return ' '.join([word for word in text.split() if len(word) >= 3])

def lemmatize_word(text: list)-> list:
    return [wordnet_lemmatizer.lemmatize(w) for w in text]

