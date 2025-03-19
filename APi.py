from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
np.random.seed(1)

# Download the tokenizer if you haven't already
nltk.download('punkt')
nltk.download('punkt_tab')


from joblib import load

word_to_index = load('Embeddings/word_to_index.joblib')

def loadModel():   
    # Registering the Custom Embedding Layer
    @tf.keras.utils.register_keras_serializable(package="CustomLayers")
    class SerializableEmbedding(Embedding):
        def get_config(self):
            config = super().get_config()
            return config

    model=tf.keras.models.load_model('Review Detector2.keras')

    return model

def convert_and_pad(tokens, max_len=510, pad_value=0, unk_index=0):
    # Convert words to indices using the dictionary, use unk_index if word not found
    indexed_tokens=[]
    for word in tokens:
        if word in word_to_index.keys():
            idx=word_to_index.get(word)
            indexed_tokens.append(idx)
    #indexed_tokens = [word_to_index.get(word, unk_index) for word in tokens]
    
    # Pad or truncate to the required length
    if len(indexed_tokens) < max_len:
        indexed_tokens += [pad_value] * (max_len - len(indexed_tokens))  # Pad
    else:
        indexed_tokens = indexed_tokens[:max_len]  # Truncate if too long
    
    return indexed_tokens

def predict_model(text):
    model=loadModel()
    sentence = text.lower()
    
    # Tokenizing the sentence
    tokens = word_tokenize(sentence)

    indexed_tokens=convert_and_pad(tokens)

    data=np.array(indexed_tokens).reshape(1,-1)
    if(np.all(data==0)):
        return ["Out of vocabulary",0]

    prediction=model.predict(data)
    prediction=prediction[0][0]
    if(prediction>0.5):
        return ["Human Written",round(prediction*100,2)]
    else:
        return ["Machine Generated",round((1-prediction)*100,2)]


app = Flask(__name__)

# Define API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review_text = data.get("review", "")

    if not review_text:
        return jsonify({"error": "No review provided"}), 400
    
    # processed_input = preprocess_text(review_text)
    
    result = predict_model(review_text)

    return jsonify({"prediction": result[0], "score": result[1]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))