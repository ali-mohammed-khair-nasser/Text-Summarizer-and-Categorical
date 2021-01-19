# Prepare libraries
from flask import Flask, render_template, request
import functions as func
import pickle
import warnings

# Stop not important warnings and define the main flask application
warnings.filterwarnings("ignore")
main_application = Flask(__name__)

# Application home page
@main_application.route("/")
def index():
    return render_template("index.html", page_title="Text Summarizer & Categorical")

# Analyze URL page
# First we get the text from the input link
# Then get classifier and the number of sentences
# Get the language for calling the right model
# Get text summary and category
@main_application.route("/analyze_url", methods=['GET', 'POST'])
def analyze_url():
    if request.method == 'POST':
        input_language = request.form['url_language']
        input_url = request.form['url_input_text']
        input_text = func.fetch_data(input_url)
        classifier_model_name = request.form['url_classifier']
        sentences_number = request.form['url_sentences_number']
        if input_language == 'english':
            classifier_model = pickle.load(open('models/en_' + classifier_model_name + '.pkl', 'rb'))
            text_summary, text_category = func.summarize_category(input_text, sentences_number, classifier_model, False)
        else:
            classifier_model = pickle.load(open('models/ar_' + classifier_model_name + '.pkl', 'rb'))
            text_summary, text_category = func.summarize_category(input_text, sentences_number, classifier_model, True)
    return render_template("index.html", page_title="Text Summarizer & Categorical", input_text=input_text, text_summary=text_summary, text_category=text_category)

# Analyze text page
# First we get the text from the input textarea
# Then get classifier and the number of sentences
# Get the language for calling the right model
# Get text summary and category
@main_application.route("/analyze_text", methods=['GET', 'POST'])
def analyze_text():
    if request.method == 'POST':
        input_language = request.form['text_language']
        input_text = request.form['text_input_text']
        classifier_model_name = request.form['text_classifier']
        sentences_number = request.form['text_sentences_number']
        if input_language == 'english':
            classifier_model = pickle.load(open('models/en_' + classifier_model_name + '.pkl', 'rb'))
            text_summary, text_category = func.summarize_category(input_text, sentences_number, classifier_model, False)
        else:
            classifier_model = pickle.load(open('models/ar_' + classifier_model_name + '.pkl', 'rb'))
            text_summary, text_category = func.summarize_category(input_text, sentences_number, classifier_model, True)
    return render_template("index.html", page_title="Text Summarizer & Categorical", input_text=input_text, text_summary=text_summary, text_category=text_category)

# Start the application on local server
if __name__ == "__main__":
    main_application.run()