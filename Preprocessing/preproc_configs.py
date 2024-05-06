from Preprocessing.helper import *


preproc_config_1 = [
    identity
]

preproc_config_2 = [
    remove_punctuation,
    remove_stopwords,
    to_lowercase,
    remove_numbers,
    remove_whitespace
]

preproc_config_nocontext = [
    replace_emoticons,
    replace_unmatched_parentheses,
    make_two_consecutive,
    replace_hashtag,
    remove_punctuation,
    decontracted,
    string_to_list,
    remove_stopwords,
    correct_text,
    lemmatize_word,
    list_to_string
]

preproc_config_context = [
    replace_emoticons,
    replace_unmatched_parentheses,
    make_two_consecutive,
    replace_hashtag,
    remove_punctuation,
    decontracted,
    string_to_list,
    correct_text,
    list_to_string
]