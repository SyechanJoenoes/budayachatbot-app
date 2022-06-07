import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model-development\chatbot_model.h5')
import json
import random
intents = json.loads(open('model-development\job_intents.json', encoding='utf-8').read())
words = pickle.load(open('model-development\words.pkl','rb'))
classes = pickle.load(open('model-development\classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# mengembalikan kumpulan kata-kata: 0 atau 1 untuk setiap kata dalam kantong yang ada dalam kalimat

def bow(sentence, words, show_details=True):
    # tokeniasi pattern yang ada di dalam intents
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matriks N kata, matriks kosakata
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # tetapkan 1 jika kata saat ini ada di posisi kosakata
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # Menyaring prediksi di bawah ambang batas
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Mengurutkan berdasarkan probabilitas yang terkuat
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "Anda Harus Menanyakan Hal yang Sesuai"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
