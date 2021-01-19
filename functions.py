# Prepare libraries and data
import nltk
import re
import heapq
import pandas as pd
from string import punctuation
punctuation = punctuation + '\n'
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.stem.isri import ISRIStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Text categories
categories = ['Economy & Business', 'Diverse News', 'Politic', 'Sport', 'Technology']

# Building the summarizer
def nltk_summarizer(input_text, number_of_sentence):
    stopWords = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(input_text):
        if word not in stopWords:
            if word not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    sentence_list = nltk.sent_tokenize(input_text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(number_of_sentence, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary

# Load English dataset
en_data = pd.read_csv(r"dataset/bbc_news_dataset.csv")
en_data = en_data.replace("entertainment", "diverse news")
en_data = en_data.replace("business", "economy & business")

# Load Arabic dataset
ar_data = pd.read_csv(r"dataset/arabic_dataset.csv")
ar_data = ar_data.replace("diverse", "diverse news")
ar_data = ar_data.replace("culture", "diverse news")
ar_data = ar_data.replace("politic", "politics")
ar_data = ar_data.replace("technology", "tech")
ar_data = ar_data.replace("economy", "economy & business")
ar_data = ar_data.replace("internationalNews", "politics")
ar_data = ar_data[~ar_data['type'].str.contains('localnews')]
ar_data = ar_data[~ar_data['type'].str.contains('society')]

# Data sterilization
# Delete links:
# This will remove all links from the text and it's include the following:
# Matches http protocols like [**http:// or https://**].
# Match optional whitespaces after http protocols.
# Optionally matches including the [**www.**] or not.
# Optionally matches whitespaces in the links.
# Matches 0 or more of one or more word characters followed by a period.
# Matches 0 or more of one or more words (or a dash or a space) followed by [**\\**].
# Any remaining path at the end of the url followed by an optional ending.
# Matches ending query params (even with white spaces, etc).
def delete_links(input_text):
    pettern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    out_text = re.sub(pettern, ' ', input_text)
    return out_text

# Fixing word lengthening:
# Word lengthening occurs when characters are wrongly repeated. English words have a max of two repeated characters like the words [**wood, school**].
# Additional characters need to ripped off, otherwise we might add misleading information.
def delete_repeated_characters(input_text):
    pattern = r'(.)\1{2,}'
    out_text = re.sub(pattern, r"\1\1", input_text)
    return out_text

# Replace spicial letters with another one
# In arabic language there is many letters can be converted to another
def replace_letters(input_text):
    replace = {"أ": "ا", "ة": "ه", "إ": "ا", "آ": "ا", "": ""}
    replace = dict((re.escape(k), v) for k, v in replace.items())
    pattern = re.compile("|".join(replace.keys()))
    out_text = pattern.sub(lambda m: replace[re.escape(m.group(0))], input_text)
    return out_text

# Delete bad symbols:
# This method removes unwanted characters from the text, such as question marks, commas, star, plus ...etc.
def clean_text(input_text):
    replace = r'[/(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛-]'
    out_text = re.sub(replace, " ", input_text)
    words = nltk.word_tokenize(out_text)
    words = [word for word in words if word.isalpha()]
    out_text = ' '.join(words)
    return out_text

# Remove arabic text vowelization
def remove_vowelization(input_text):
    vowelization = re.compile(""" [ًٌٍَُِّْـ]""", re.VERBOSE)
    out_text = re.sub(vowelization, '', input_text)
    return out_text

# Delete stopwords:
# Like prepositions and hyphens words. for example [**and, in, or ...etc**].
def delete_stopwords(input_text):
    stop_words = set(nltk.corpus.stopwords.words("arabic") + nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    wnl = nltk.WordNetLemmatizer()
    lemmatizedTokens = [wnl.lemmatize(t) for t in tokens]
    out_text = [w for w in lemmatizedTokens if w not in stop_words]
    out_text = ' '.join(out_text)
    return out_text

# For arabic text only becouse it will give us better results when we train the models
def stem_text(input_text):
    st = ISRIStemmer()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(input_text)
    out_text = [st.stem(w) for w in tokens]
    out_text = ' '.join(out_text)
    return out_text

# Text prepare:
# Apply all previous functions to sterilize the input text.
# Convert letters to lowercase to make all words in the text in the same letters sensitivity.
def text_prepare(input_text, ar_text):
    out_text = delete_links(input_text)
    out_text = delete_repeated_characters(out_text)
    out_text = clean_text(out_text)
    out_text = delete_stopwords(out_text)
    if ar_text:
        out_text = replace_letters(out_text)
        out_text = remove_vowelization(out_text)
        out_text = stem_text(out_text)
    else:
        out_text = out_text.lower()
    return out_text

# Apply text prepare function for english and arabic dataset
en_data['Processed Text'] = en_data['Text'].apply(text_prepare, args=(False,))
ar_data['Processed Text'] = ar_data['text'].apply(text_prepare, args=(True,))

# Label encoder
# Convert labels to numeric values so the machine learning models can deal with it
en_label_encoder = LabelEncoder()
en_data['Category Encoded'] = en_label_encoder.fit_transform(en_data['Category'])

# After label encoding we sholud change some labels to another becouse the arabic dataset labels is not the same with english dataset
ar_label_encoder = LabelEncoder()
ar_data['Category Encoded'] = ar_label_encoder.fit_transform(ar_data['type'])
ar_data['Category Encoded'] = ar_data['Category Encoded'].replace(1, 0)
ar_data['Category Encoded'] = ar_data['Category Encoded'].replace(0, 1)

# Spliiting the data to train and test
# 80% of the data used for models train
# 20% of the data used for test and validation
en_X_train, en_X_test, en_y_train, en_y_test = train_test_split(en_data['Processed Text'], en_data['Category Encoded'], test_size=0.2, random_state=0)
ar_X_train, ar_X_test, ar_y_train, ar_y_test = train_test_split(ar_data['Processed Text'], ar_data['Category Encoded'], test_size=0.2, random_state=0)

# TF-IDF vectorizer:
# The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora.
# It helps to penalize too frequent words and provide better features space. 
def tfidf_features(X_train, X_test, ngram_range):
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, ngram_range))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test

# Summarize and predict for input text:
def summarize_category(input_text, statements, model_name, ar_text=False):
    statements = int(statements)
    summary_text = nltk_summarizer(input_text, statements)
    input_text_arr = [text_prepare(input_text, ar_text)]
    if ar_text:
        features_train, features_test = tfidf_features(ar_X_train, input_text_arr, 2)
    else:
        features_train, features_test = tfidf_features(en_X_train, input_text_arr, 2)
    text_prediction = model_name.predict(features_test.toarray())
    text_category = categories[text_prediction[0]]
    return summary_text, text_category

# Fetch data from URL function:
def fetch_data(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text