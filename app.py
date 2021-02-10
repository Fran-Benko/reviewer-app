from flask import Flask, render_template, url_for, request, render_template_string
import pickle
import numpy as np
import nltk as nlp
from textblob import TextBlob
import re
from keras.models import load_model

'''
Ejecutar en consola python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
'''

tfidf_vectorizer = pickle.load(open('trained_vectorizer_tfidf_1.pickle', 'rb'))
pca_trained = pickle.load(open('trained_pca_uni_tfidf.pickle', 'rb'))
model_trained = load_model('model_analize_reviewer.h5')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model')
def model():
    return render_template('model.html', data=0)

@app.route('/data')
def data():
    return render_template('data.html')    

@app.route('/index')
def index2():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    '''
    Entrada: Entra la review
    '''
    text_i = request.form['text_box']
    text = str(text_i)

    '''
    Limpieza de la review
    '''
    text = re.sub("n\'t"," not",text)
    text = text.lower() 
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    
    '''
    Preparacion para inferencia
    '''
    vectorized_text = tfidf_vectorizer.transform(np.array([text]))

    polaridad = TextBlob(text).sentiment.polarity
    subjetividad = TextBlob(text).sentiment.subjectivity

    # Creamos una matriz aumentada para sumarle las features de polarity y subjectivity
    mat_aumentada = np.zeros((vectorized_text.shape[0],vectorized_text.shape[1]+2))
    mat_aumentada[:,:-2] = vectorized_text.toarray()
    mat_aumentada[:,-2] = polaridad
    mat_aumentada[:,-1] = subjetividad

    X_text = pca_trained.transform(mat_aumentada)

    pred = model_trained.predict_classes(X_text) + 1
    
    return render_template('model.html', data=pred, text=text_i)


if __name__ == '__main__':
    app.run()