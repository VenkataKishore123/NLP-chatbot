import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
import json
import numpy as np
import pickle
import tensorflow as tf

# Load the necessary files
lemmatizer = WordNetLemmatizer()
data_file = open('data.json').read()
intents = json.loads(data_file)
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Prepare test data in the same way as training data
test_data_file = open('data.json').read()
test_intents = json.loads(test_data_file)

test_documents = []
for intent in test_intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        test_documents.append((w, intent['tag']))

test_x = []
test_y = []
output_empty = [0] * len(classes)

for doc in test_documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    test_x.append(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    test_y.append(output_row)

test_x = np.array(test_x)
test_y = np.array(test_y)

# Make predictions
pred_y = model.predict(test_x)
pred_classes = np.argmax(pred_y, axis=1)
true_classes = np.argmax(test_y, axis=1)

# Generate classification report
report = classification_report(true_classes, pred_classes, target_names=classes, digits=2)
print(report)

# Optionally, save the report to a file
with open('classification_report.txt', 'w') as f:
    f.write(report)
