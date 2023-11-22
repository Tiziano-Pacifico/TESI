import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
from gensim.corpora.dictionary import Dictionary
import spacy
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def preprocess_text(text):
    doc = nlp(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

path = "C:\\Users\\Tiziano Pacifico\\Desktop\\TESI\\RedditDS\\chatGPT\\LDA_full_out\\"
filename = "cleaned_full_blobs.txt"

print("Opening file")
blobs = []
with open(path + filename, 'r') as file:
    for riga in file:
        blob = [str(elemento) for elemento in riga.strip().split()]
        blobs.append(blob)

print("File successfully opened")

print("Initializing LDA")
dictionary = Dictionary(blobs)
corpus = [dictionary.doc2bow(text) for text in blobs]
        
print("Start LDA coherence analysis\n\n")
coherence_list = []
for i in range(95,100):
    if __name__ == '__main__': 
        lda_model = LdaMulticore(corpus, num_topics=i, id2word=dictionary, passes=10, workers=4)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=blobs, dictionary=dictionary, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f"Numero di topic: {i} -- coherence: {coherence_lda}")
        coherence_list.append(coherence_lda)

filename = "LDA_coherence_list.txt"

with open(path+filename, 'w') as file:
    for elemento in coherence_list:
            file.write(str(elemento) + '\n')