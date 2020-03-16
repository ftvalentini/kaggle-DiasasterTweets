import re
import string
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import stop_words
# from unidecode import unidecode

def remove_bad_characters(chars):
    """ removes weird characters from string
    such as: \x89Ûª, \x89Û¢åÊ, åÇ, etc.
    """
    clean_string = chars.encode('ascii', errors='ignore').decode('utf-8')
    return clean_string

def full_clean_text(chars):
    """ clean string (document) before tokenizing and fitting simple model
    """
    # remove weird characters
    chars = remove_bad_characters(chars)
    # remove links
    chars = re.sub(r'https?:\/\/.*[\r\n]*', '', chars)
    # remove hashtags
    chars = re.sub(r'#\S+', '', chars)
    # remove #
    chars = re.sub(r'#', ' ', chars)
    # remove mentions
    chars = re.sub(r'@\S+', '', chars)
    # remove punctuation
    chars = "".join([char for char in chars if char not in string.punctuation])
    # remove some punctuation chars
    chars = re.sub(r'[-+_+¡+¿+]',' ',chars)
    # remove numbers
    chars = re.sub(r'\d+',' ',chars)
    # replace accentuated chars for chars without accents
    match_replace = [('á|Á','a'),('é|É','e'),('í|Í','i'),('ó|Ó','o'),('ú|Ú','u')]
    for i in range(len(match_replace)):
        chars = re.sub(match_replace[i][0], match_replace[i][1], chars)
    # remove duplicated whitespace
    chars = re.sub(' +', ' ', chars)
    return chars.strip()

def tokeniza(chars, keyword=None):
    """
    Tokenize a string (duplicates keywords if any)
    """
    tokenizer = TweetTokenizer(preserve_case=False
                               ,strip_handles=True,reduce_len=True)
    tokens = tokenizer.tokenize(chars)
    return tokens

def get_stopwords():
    """ Deprecated: it's better to use 'english' default in sklearn vectorizer
    """
    sw1 = stopwords.words('english')
    # sw2 = [unidecode(w) for w in sw1]
    # sw = list(set(sw1 + sw2))
    return sw1
