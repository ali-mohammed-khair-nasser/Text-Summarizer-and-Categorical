# Text Summarizer & Categorical &middot; [![GitHub](https://img.shields.io/github/license/ali-mohamed-nasser/Text-Summarizer-and-Categorical)](https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/LICENSE) ![GitHub top language](https://img.shields.io/github/languages/top/ali-mohamed-nasser/Text-Summarizer-and-Categorical) [![Codacy grade](https://img.shields.io/codacy/grade/cd25af7df2854c4588f4e0439223e708)](https://www.codacy.com/gh/ali-mohamed-nasser/Text-Summarizer-and-Categorical/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ali-mohamed-nasser/Text-Summarizer-and-Categorical&amp;utm_campaign=Badge_Grade)

### Introduction
Summarization is the task of condensing a piece of text to a shorter version, reducing the size of the initial text while at the same time preserving key informational elements and the meaning of content. Since manual text summarization is a time expensive and generally laborious task, the automatization of the task is gaining increasing popularity and therefore constitutes a strong motivation for academic research. Automatic text summarization is a common problem in machine learning and natural language processing (NLP).

### Natural Language Processing (NLP)
Natural Language Processing is the technology used to aid computers to understand the human’s natural language. It’s not an easy task teaching machines to understand how we communicate. Natural Language Processing which usually shortened as NLP, is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable.

### Graphical User Interface
The graphical user interface (GUI) is the point of human-computer interaction and communication in a device. This can include display screens, keyboards, a mouse and the appearance of a desktop. It is also the way through which a user interacts with an application or a website. The GUI of this application builded using web languages like ``` HTML ``` ``` CSS ``` ``` javascript ``` and this is how it look like:

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/ui.svg" width="1200">

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
- A new window should open, showing the NLTK downloader. Next, select the packages or collections you want to download. And for more information see the documentation from [nltk.org](https://www.nltk.org/data.html)

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
After reading the input text and the number of sentences from the user and getting the model name and the language, then pass that information to the main function that apply the text summarization and get the right category and for each input text we do the following steps.

# Getting Text Summary 
In this step will apply text tokenization which is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. Along with text classifcation, in text mining, it is necessay to incorporate a parser in the pipeline which performs the tokenization of the documents. And after text tokenization will check each token that not in the english or arabic stopwords (like the words: and, or, then, etc) and not in the punctuation (like: %, #, $, etc) then calculate the frequencies for each token. 

After this step we normalize all tokens by diving that frequencies on the maximum one using:
```python
maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)
```
The next step is to calculate the sentences scores for each sentence so we split the input text again into sentences using ``` nltk.sent_tokenize() ``` function and will calculate that scores in the same way in the frequencies calculation. Finnaly will use ``` heapq.nlargest() ``` function and this function will arrange the sentences in descending order and take the required number of sentences and join it to create the summary.

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/summarization.png" width="1200">

# Getting Text Category
Here we have alot of works to do for extract the features from that text for use that features in classification algorithms. And in this section, we start to talk about text cleaning since most of urls or texts contain a lot of noise, and after that we will pass the cleaned text to TF-IDF vectorizer to prepare it to classification model. Getting cate gory process explained in the following diagram:

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/get_category.png" width="1200">

### Text Cleaning and Pre-processing
In Natural Language Processing (NLP), most of the text and documents contain many words that are redundant for text classification, such as stopwords, miss-spellings, slangs, and etc. In this section, we briefly explain some techniques and methods for text cleaning and pre-processing text documents. In many algorithms like statistical and probabilistic learning methods, noise and unnecessary features can negatively affect the overall perfomance. So, elimination of these features are extremely important. We will do multi steps to clear the input text and this steps is:

#### Delete Links
This will remove all links from the text and it's include the following:
- Matches http protocols like ``` http:// ``` or ``` https:// ```
- Match optional whitespaces after http protocols.
- Optionally matches including the ``` www. ``` or not.
- Optionally matches whitespaces in the links.
- Matches 0 or more of one or more word characters followed by a period.
- Matches 0 or more of one or more words (or a dash or a space) followed by ```\ ```
- Any remaining path at the end of the url followed by an optional ending.
- Matches ending query params (even with white spaces, etc).

#### Fixing Word Lengthening
- Word lengthening occurs when characters are wrongly repeated. English and arabic words have a max of two repeated characters like the words ``` wood, school ``` in english and ``` مؤسسة ``` in arabic.
- Additional characters need to ripped off, otherwise we might add misleading information.
- Replace spicial letters with another one. In arabic language there is many letters can be converted to another like the letters ``` أ ,ا ,ة ,ه ,إ ,آ ```

#### Delete Bad Symbols & Stopwords
Another issue of text cleaning as a pre-processing step is noise removal. Text documents generally contains characters like punctuations or special characters and they are not necessary for text mining or classification purposes. Although punctuation is critical to understand the meaning of the sentence, but it can affect the classification algorithms negatively.
- Example of bad symbols ``` /(){}\[\]|@âÂ,;\?\'\"\*…؟–’،!&\+-:؛- ```
- Also remove arabic text vowelization.
- Stopwords like prepositions and hyphens words. for example ``` and, in, or, etc ```

#### Text Stemming
Text Stemming is modifying a word to obtain its variants using different linguistic processeses like affixation (addition of affixes). Will apply for arabic text only becouse it will give us better results when we train the models. For example, the stem of the word ``` التعليمية ``` is ``` علم ```

#### Convert to Lowercase 
Sentences can contain a mixture of uppercase and lower case letters. Multiple sentences make up a text document. To reduce the problem space, the most common approach is to reduce everything to lower case. This brings all words in a document in same space, but it often changes the meaning of some words, such as "US" to "us" where first one represents the United States of America and second one is a pronoun. To solve this, slang and abbreviation converters can be applied.

### Get TF-IDF features
The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. Here we conver our text to numiric values so the AI models can deal with it. And this technique helps to penalize too frequent words and provide better features space. 
The advantages of this feature extraction technique is:
- Easy to compute.
- Easy to compute the similarity between 2 documents using it.
- Basic metric to extract the most descriptive terms in a document.
- Common words do not affect the results due to IDF ``` am, is, etc ```

And here is a simple example of converting the words in each sentences in the dataset or the input text into a numiric values using the TF-IDF vectorizer:

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/tf_idf.png" width="1200">

### Predict The Right Category Using Pre-trained Models
I did trained 6 different models for both arabic and english datasets and all of this models built-in in the sklearn library, and this models give me a heigh accuracy score for the data classification. There is a simple comparison between this models:

#### 1. Decision Tree
One of the earlier classification algorithms for text and data mining is a decision tree. Decision tree classifiers (DTC's) are used successfully in many diverse areas of classification. The structure of this technique includes a hierarchical decomposition of the data space (only train dataset). Decision tree as classification task was introduced by D. Morgan and developed by JR. Quinlan. The main idea is creating trees based on the attributes of the data points, but the challenge is determining which attribute should be at the parent level and which one should be at the child level. To solve this problem, De Mantaras introduced statistical modeling for feature selection in tree. For more information about decision tree click [here](https://scikit-learn.org/stable/modules/tree.html)

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/tree.png" width="1200">

#### 2. Random Forest
Random forests or random decision forests technique is an ensemble learning method for text classification. This method was introduced by T. Kam Ho in 1995 for the first time which used t trees in parallel. This technique was later developed by L. Breiman in 1999 that they found converged for RF as a margin measure. For more information about random forest click [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/random_forest.png" width="1200">

#### 3. Ridge Classifier
Ridge regression is a way to create a parsimonious model when the number of predictor variables in a set exceeds the number of observations, or when a data set has multicollinearity (correlations between predictor variables). Tikhivov’s method is basically the same as ridge regression, except that Tikhonov’s has a larger set. It can produce solutions even when your data set contains a lot of statistical noise (unexplained variation in a sample). For more information about ridge classifier click [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)

#### 4. Logistic Regression
Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval, or ratio-level independent variables. For more information about logistic regression click [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/linear.png" width="1200">

#### 5. K-Nearest Neighbors
The k-nearest neighbor's algorithm (KNN) is a non-parametric technique used for classification. This method is used in Natural-language processing (NLP) as a text classification technique in many kinds of research in the past decades. For more information about KNN click [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

<img src="https://github.com/ali-mohamed-nasser/Text-Summarizer-and-Categorical/blob/main/images/knn.png" width="1200">

#### 6. Gaussian Naive Bayes
One assumption used in the (NLP) is the strong independence assumptions between the features. These classifiers assume that the value of a particular feature is independent of the value of any other feature. In a supervised learning situation, Naive Bayes Classifiers are trained very efficiently. For more information about Gaussian NB click [here](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

### Compare Models Results On The Used Data
Model Name | Accuracy on English Data | Accuracy on Arabic Data
:-------------------------:|:-------------------------:|:-------------------------:
Decision Tree | 87.9% | 80.9%
Random Forest | 94.3% | 86.7%
Ridge Classifier | 98.3% | 93.1%
Logistic Regression | 98.9% | 92.1%
K-Nearest Neighbors | 95.9% | 91.3%
Gaussian Naive Bayes | 94.6% | 88.2%

# References
Here is a list of sources I used to build this project:
- Dataset used to train the models from [kaggle](https://www.kaggle.com/). No need to download the datasets, it provided with this project.
- NLTK tutorial book from [nltk.org](https://www.nltk.org/book/)
- Text Preprocessing for NLP (Natural Language Processing) in [medium.com](https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95)
- [Building a Text Summarizer Flask App](https://www.youtube.com/watch?v=xvLQdP549NA&list=PLJ39kWiJXSiyAFG2W3CUPWaLhvR5CQmTd&index=9). I watched 70% of this video.
- Text Classification Algorithms on [medium.com](https://medium.com/text-classification-algorithms/text-classification-algorithms-a-survey-a215b7ab7e2d)
- News classification tutorial on [youtube.com](https://www.youtube.com/watch?v=dtK7Xhn8XjY)
- Scikit learn tutorials from [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
- Flask Tutorials playlist [here](https://www.youtube.com/watch?v=mqhxxeeTbu0&list=PLzMcBGfZo4-n4vJJybUVV3Un_NFS5EOgX)
