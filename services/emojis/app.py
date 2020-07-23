#from keras.models import model_from_json
#import keras
from tensorflow.keras.models import model_from_json
import emoji
import pandas as pd 
import numpy as np 
#import dill as pickle


emoji_dictionary = {"0": ":u2764uFE0F:",    
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:"
                   }

with open("services/emojis/model.json","r") as file:
	model = model_from_json(file.read())

model.load_weights("services/emojis/model.h5")


#model = pickle.load(open("services/emojis/model1.pk", 'rb'))


#model._make_predict_function()




# wordToIndex=pickle.load(open('wordToindex.pkl','rb'))

embeddings = {}
with open('services/emojis/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs
f.close()


def getOutputEmbeddings(X):
    
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
            
    return embedding_matrix_output

def predict(x):
	X = pd.Series([x])
	emb_x = getOutputEmbeddings(X)
	p = model.predict(emb_x)
	return emoji.emojize(emoji_dictionary[str(np.argmax(p))])





