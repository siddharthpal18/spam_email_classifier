#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip3 install tensorflow


# In[1]:


import re
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sqlite3
import spacy
import nltk
import pickle
import plotly

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from spacy.util import compounding
from spacy.util import minibatch
from collections import defaultdict
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.layers import (LSTM,
                          BatchNormalization,
                          Dense, 
                          TimeDistributed,
                          Dropout,
                          Bidirectional,
                          Flatten, 
                          Embedding,
                          GlobalMaxPool1D)

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
    accuracy_score
)


# In[3]:


df = pd.read_csv('spam.csv', encoding='latin-1')
df


# In[4]:


df = df.dropna(how="any", axis=1)
df.columns = ['target', 'message']

df.head()


# In[5]:


import sqlite3
import pandas as pd

def create_database_table(database_name, table_name):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    target TEXT,
                    message TEXT
                )''')
    conn.commit()
    conn.close()

def upload_data_to_database(csv_file, database_name, table_name, encoding='utf-8'):
    # Read CSV file into DataFrame with specified encoding
    df = pd.read_csv(csv_file, encoding=encoding)

    # Drop rows with any missing values and rename columns
    df = df.dropna(how="any", axis=1)
    df.columns = ['target', 'message']

    # Connect to SQLite database
    conn = sqlite3.connect(database_name)

    # Write DataFrame to SQLite database
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Close database connection
    conn.close()


if __name__ == "__main__":
    # CSV file containing the data
    csv_file = 'spam.csv'

    # Name of the SQLite database file
    database_name = 'mydata.db'

    # Name of the table in the database
    table_name = 'messages'

    # Create database table
    create_database_table(database_name, table_name)

    # Upload data to database
    upload_data_to_database(csv_file, database_name, table_name, encoding='latin1')

    print("Data uploaded successfully to SQLite database.")


## Fetch data from database

def fetch_data_from_database(database_name, table_name):
    # Connect to SQLite database
    conn = sqlite3.connect(database_name)

    # Execute a SELECT query to fetch data from the table
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # Close connection
    conn.close()

    return df

if __name__ == "__main__":
    # Name of the SQLite database file
    database_name = 'mydata.db'

    # Name of the table in the database
    table_name = 'messages'

    # Fetch data from database
    df = fetch_data_from_database(database_name, table_name)

    # Print the first few rows of the DataFrame
    print(df.head())


# In[7]:


df['message_len'] = df['message'].apply(lambda x: len(x.split(" ")))


# In[8]:


df


# In[9]:


max(df['message_len'])


# # Checking either the dataset is distributed or not

# In[10]:


balance_counts = df.groupby('target')['target'].agg('count').values


# In[11]:


plt.pie(balance_counts, explode=(0, 0.06),labels=df['target'].unique(),autopct='%.1f%%')
plt.show()


# In[12]:


# This is an imbalanced dataset


# # 2. Data Preprocessing

# In[13]:


def clean_text(text):
    '''Lowering each text case
    Removing text in square brackets
    Removing Links
    Removing Punctuations
    Removing words containing numbers'''
    
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' %re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[14]:


df['message_clean'] = df['message'].apply(clean_text)
df.head()


# In[15]:


stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text
    
df['message_clean'] = df['message_clean'].apply(remove_stopwords)
df.head()


# In[16]:


stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


# In[17]:


df['message_clean'] = df['message_clean'].apply(stemm_text)
df.head()


# In[18]:


def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    
    return text


# In[19]:


df['message_clean'] = df['message_clean'].apply(preprocess_data)
df.head()


# In[20]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['target'])

df['target_encoded'] = le.transform(df['target'])
df.head()

# Save the LabelEncoder to a file for later use in Flask app
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)


# In[21]:


wc = WordCloud(
    background_color='white', 
    max_words=200, 
    
)
wc.generate(' '.join(text for text in df.loc[df['target'] == 'ham', 'message_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for HAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[ ]:





# In[22]:


wc = WordCloud(
    background_color='white', 
    max_words=200, 
    
)
wc.generate(' '.join(text for text in df.loc[df['target'] == 'spam', 'message_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for SPAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[23]:


## Vectorization


# In[24]:


# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
x = df['message_clean']
y = df['target_encoded']

print(len(x), len(y))

label_encoder = LabelEncoder()
df['target_encoded'] = label_encoder.fit_transform(df['target'])

# In[25]:


# Split into train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer

# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(x_train)

# Step 2: Save the vectorizer using pickle
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vect, vectorizer_file)
# In[27]:


# Use the trained to create a document-term matrix from train and test sets
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)


# In[28]:


vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)


# In[29]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(x_train_dtm)
x_train_tfidf = tfidf_transformer.transform(x_train_dtm)

x_train_tfidf


# In[30]:


## Word embeddings glove


# In[31]:


texts = df['message_clean']
target = df['target_encoded']


# In[32]:


# Calculate the length of our vocabulary
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(texts)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length


# In[33]:


def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(texts), 
    length_long_sentence, 
    padding='post'
)

train_padded_sentences


# In[34]:


embeddings_dictionary = dict()
embedding_dim = 100

# Load GloVe 100D embeddings
with open("glove.txt","r+", encoding="utf-8") as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions

# embeddings_dictionary


# In[35]:


# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.

embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
embedding_matrix


# In[36]:


import plotly.figure_factory as ff

x_axes = ['Ham', 'Spam']
y_axes =  ['Spam', 'Ham']

def conf_matrix(z, x=x_axes, y=y_axes):
    
    z = np.flip(z, 0)

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                      xaxis = dict(title='Predicted value'),
                      yaxis = dict(title='Real value')
                     )

    # add colorbar
    fig['data'][0]['showscale'] = True
    
    return fig


# In[37]:


# Create a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Train the model
nb.fit(x_train_dtm, y_train)


# In[38]:


# Make class anf probability predictions
# Make class predictions
y_pred_class = nb.predict(x_test_dtm)

# Convert numeric labels back to text labels
y_pred_labels = le.inverse_transform(y_pred_class)
from sklearn.preprocessing import LabelEncoder

# Fit LabelEncoder on the target labels
le.fit(y_test)



# Calculate accuracy using the encoded predictions
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# Print classification report
print(classification_report(y_test, y_pred_class))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred_class))

y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]


# In[39]:


# calculate accuracy of class predictions
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))         


# In[40]:


# Calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


# In[41]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), 
                 ('tfid', TfidfTransformer()),  
                 ('model', MultinomialNB())])  


# In[42]:


pipe.fit(x_train, y_train)

y_pred_class = pipe.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred_class))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))


# In[43]:


## XG Boost


# In[44]:


import xgboost as xgb

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=80,
        use_label_encoder=False,
        eval_metric='auc',
        # colsample_bytree=0.8,
        # subsample=0.7,
        # min_child_weight=5,
    ))
])


# In[45]:


# Fit the pipeline with the data
pipe.fit(x_train, y_train)

y_pred_class = pipe.predict(x_test)
y_pred_train = pipe.predict(x_train)

print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))


# In[47]:


pickle.dump(nb, open('model.pkl', 'wb'))





