import streamlit as st
import tensorflow_text as text
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import pickle
import datetime
import os.path


st.title("Spam Mail Classification")
st.subheader("An NLP model on top of BERT to classify spam mail")

if not os.path.isfile('model.sav'):
    # load data
    df_first = pd.read_csv('./spam_data.csv')
    # st.write('df_first.head()',df_first.head())
    # rearrange the columns
    df_first.drop(columns=df_first.columns[0], axis=1, inplace=True)
    # st.write('df_first.head()', df_first.head())
    # rename columns
    df_first = df_first.rename({'target': 'Category', 'text': 'Message'}, axis=1)
    # st.write('df_first.head()', df_first.head())
    # check count and unique and top values and their frequency
    df_first['Category'].value_counts()
    # check percentange of data - states how much data needs to be balanced
    # st.write("str(round(747/4825,2))+'%'", str(round(747/4825,2))+'%')
    # creating 2 new dataframe as df_hm , df_spm

    df_spm = df_first[df_first['Category']=='spam']
    # st.write("Spam Dataset Shape:", df_spm.shape)

    df_hm = df_first[df_first['Category']=='ham']
    # st.write("Ham Dataset Shape:", df_hm.shape)
    # downsampling ham dataset - take only random 747 example
    # will use df_spm.shape[0] - 747

    df_hm_downsampled = df_hm.sample(df_spm.shape[0])
    df_hm_downsampled.shape
    # concating both dataset - df_spm and df_hm_balanced to create df_balanced dataset
    df_balanced = pd.concat([df_spm , df_hm_downsampled])
    df_balanced.head()
    df_balanced['Category'].value_counts()
    df_balanced.sample(10)
    # creating numerical repersentation of category - one hot encoding
    df_balanced['spam'] = df_balanced['Category'].apply(lambda x:1 if x=='spam' else 0)
    # displaying data - spam -1 , ham-0
    df_balanced.sample(4)
    # loading train test split
    X_train, X_test , y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'],
                                                        stratify = df_balanced['spam'])
    # check for startification
    # st.write('y_train.value_counts()', y_train.value_counts())
    # 560/560
    # st.write('y_test.value_counts()', y_test.value_counts())
    # 187/187
    # downloading preprocessing files and model
    bert_pre_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    bert_enc_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
    bert_preprocessor = hub.KerasLayer(bert_pre_url)
    bert_encoder = hub.KerasLayer(bert_enc_url)
    text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
    preprocessed_text = bert_preprocessor(text_input)
    embeed = bert_encoder(preprocessed_text)
    dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embeed['pooled_output'])
    outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense')(dropout)

    # creating final model
    model = tf.keras.Model(inputs = [text_input], outputs = [outputs])
    # check summary of model
    # st.write('model.summary()', model.summary())
    Metrics = [tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
            ]

    model.compile(optimizer ='adam',
                loss = 'binary_crossentropy',
                metrics = Metrics)
    # #@title Optional 
    # # optional - defining tensorflow callbacks
    # import tensorflow as tf
    # %load_ext tensorboard
    # !rm -rf ./logs/
    shutil.rmtree('./logs/')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # # %tensorboard --logdir logs/fit
    with tf.device('/cpu:0'):
        history = model.fit(X_train, y_train, epochs = 1 , callbacks = [tensorboard_callback])
    with tf.device('/cpu:0'):
        # Evaluating performace
        model.evaluate(X_test,y_test)
    # save the model into the directory
    pickle.dump(model, open('model.sav', 'wb'))
else:
    print('MODEL READEDDDD')
    # read the model if saved
    model = pickle.load(open('model.sav', 'rb'))

# with tf.device('/cpu:0'):
#     # getting y_pred by predicting over X_text and flattening it
#     y_pred = model.predict(X_test)
#     y_pred = y_pred.flatten() # require to be in one dimensional array , for easy maniputation
# # checking the results y_pred
# y_pred = np.where(y_pred>0.5,1,0 )
# st.write('y_pred', y_pred)
# # importing consfusion maxtrix
# # creating confusion matrix 
# cm = confusion_matrix(y_test,y_pred)
# st.write('cm', cm)
# # plotting as graph - importing seaborn
# # creating a graph out of confusion matrix
# st.write(sns.heatmap(cm, annot = True, fmt = 'd'))
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# # printing classification report
# st.write(classification_report(y_test , y_pred))
plain_text = st.text_input('Enter your mail body to predict')
if plain_text:
    text_to_predict = [plain_text]
    predict_text = [
                    # Spam
                    'We’d all like to get a $10,000 deposit on our bank accounts out of the blue, but winning a prize—especially if you’ve never entered a contest', 
                    'Netflix is sending you a refund of $12.99. Please reply with your bank account and routing number to verify and get your refund', 
                    'Your account is temporarily frozen. Please log in to to secure your account ', 

                    #ham
                    'The article was published on 18th August itself',
                    'Although we are unable to give you an exact time-frame at the moment, I would request you to stay tuned for any updates.',
                    'The image you sent is a UI bug, I can check that your article is marked as regular and is not in the monetization program.'
    ]
    with tf.device('/cpu:0'):
        test_results = model.predict(text_to_predict)
    output = np.where(test_results>0.5,'spam', 'ham')
    output = 'Spam' if test_results>0.5 else 'Ham'
    st.write(f'This mail is likely to be a {output}')
    st.write(f'Spam percentage is {test_results[0][0]}')
    # def get_embedding(sentence_arr):
    #     'takes in sentence array and return embedding vector'
    #     preprocessed_text = bert_preprocessor(sentence_arr)
    #     embeddings = bert_encoder(preprocessed_text)['pooled_output']
    #     return embeddings
    # with tf.device('/cpu:0'):
    #     e = get_embedding([
    #                 'We’d all like to get a $10,000 deposit on our bank accounts out of the blue, but winning a prize—especially if you’ve never entered a contest',
    #                 'The image you sent is a UI bug, I can check that your article is marked as regular and is not in the monetization program.'
    #     ])
    # # check similarity score
    # st.write(f'Similarity score between 1st sentence(spam) and second sentence(spam) : {cosine_similarity([e[0]] , [e[1]])}')

