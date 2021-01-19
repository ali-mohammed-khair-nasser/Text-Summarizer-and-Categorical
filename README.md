# Text Summarizer & Categorical
### Introduction
Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the initial text while at the same time preserving key informational elements and the meaning of content. Since manual text summarization is a time expensive and generally laborious task, the automatization of the task is gaining increasing popularity and therefore constitutes a strong motivation for academic research. Automatic text summarization is a common problem in machine learning and natural language processing (NLP).

### Natural Language Processing (NLP)
Natural Language Processing is the technology used to aid computers to understand the human’s natural language. It’s not an easy task teaching machines to understand how we communicate. Natural Language Processing which usually shortened as NLP, is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable.

# Knowledges required for this project
### Python programming language
Basic knowledge of coding and syntax formats of python is essential for this project. Python is advantageous as it easy to comprehend and consists of large number of inbult libraries that facilitate faster outputs.
- Learn python through tutorials from [python.org](https://www.python.org/about/gettingstarted/)
- Download python from [python.org](https://www.python.org/downloads/)

### NLTK Library
NLTK is a leading platform for building Python programs to work with human language data. t provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.
- Learn NLTK through book from [nltk.org](https://www.nltk.org/book/)
- Install NLTK using pip ``` pip install nltk ```
- Install NLTK in anaconda environment ``` conda install -c anaconda nltk ```
- To installing NLTK data run the python interpreter or new Jupiter notebook in anaconda then type the following commands:
``` import nltk ``` then ``` nltk.download() ```
- A new window should open, showing the NLTK downloader. Next, select the packages or collections you want to download. And for more information see the documentation from [here](https://www.nltk.org/data.html)

### Pandas Library
In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
- Learn Pandas from [pandas.pydata.org](https://pandas.pydata.org/docs/)
- Install Pandas using pip ``` pip install pandas ```
- Install Pandas in anaconda environment ``` conda install -c anaconda pandas ```

### Urllib & Beautifulsoup4 Libraries
This two libraries help in working with URL and fetching data from external websites to add it to the main program for the summarize and categorize.
- Install urllib ``` pip install urllib3 ```
- Install beautifulsoup4 ``` pip install beautifulsoup4 ```

### Scikit Learn
Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
- Learn Scikit-learn from [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
- Install Scikit-learn using pip ``` pip install scikit-learn ```
- Install Scikit-learn in anaconda environment ``` conda install -c anaconda scikit-learn ```
- To save and load models you need to install pickle ``` pip install pickle-mixin ```

### Flask Framework
Flask is a popular Python web framework, meaning it is a third-party Python library used for developing web applications.
- Install Flask on your machine ``` pip install Flask ```


# Usage
To run this project make sure you have the required installation of Python, NLTK with it's data, urllib, bs4, pandas, Scikit-learn, and Flask framework then follow the steps given below:
- Clone or download this repository ``` https://github.com/ali-mohamed-nasser/Text-Summarizer-Categorical.git ```
- Run the ``` main.py ``` file to start Flask server and use the application.
- You don't need to train classification models on your own. I have trained all model and saved it in the dictionary ``` models ```, but you have the datasets in ``` dataset ``` directory and the Jupiter notebook ``` NLTK Summarizer.ipynb ``` so you can retrain it if you want.

# How does it work?
After reading the input text and the number of sentences from the user and getting the model name and the language, then pass that information to the main function that apply the text summarization and get the right category and for each input text we do the following steps:

### Getting Text Summary 
In this step will apply text tokenization which is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. Along with text classifcation, in text mining, it is necessay to incorporate a parser in the pipeline which performs the tokenization of the documents. And after text tokenization will check each token that not in the english or arabic stopwords (like the words: and, or, then, etc) and not in the punctuation (like: %, #, $, etc) then calculate the frequencies for each token. 

After this step we normalize all tokens by diving that frequencies on the maximum one using:
```python
maximum_frequncy = max(word_frequencies.values())
for word in word_frequencies.keys():
  word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)
```
The next step is to calculate the sentences scores for each sentence so we split the input text again into sentences using ``` nltk.sent_tokenize() ``` function and will calculate that scores in the same way in the frequencies calculation. Finnaly will use ``` heapq.nlargest() ``` function and this function will arrange the sentences in descending order and take the required number of sentences and join it to create the summary.
