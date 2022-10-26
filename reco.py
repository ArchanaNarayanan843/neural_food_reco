import streamlit as st
import pickle
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
from zipfile import ZipFile
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model, Sequential
from pathlib import Path
import matplotlib.pyplot as plt

#option=pickle.load(open('user.pkl','rb'))
#opt=pd.DataFrame(option)


#selected_item_name=st.selectbox(
#'Which item do you prefer?',opt['user_name'].values)
user_list=pickle.load(open('user.pkl','rb'))
user=pd.DataFrame(user_list)

data_list=pickle.load(open('data.pkl','rb'))
data=pd.DataFrame(data_list)

item_list=pickle.load(open('item_df.pkl','rb'))
item=pd.DataFrame(item_list)

image=pickle.load(open('image.pkl','rb'))
img=pd.DataFrame(image)

users=data['user_name'].unique().tolist()
user2user_encoded={x:i for i, x in enumerate(users)}
userencoded2user={i:x for i, x in enumerate(users)}

items=data['item_id'].unique().tolist()
item2item_encoded={x:i for i, x in enumerate(items)}
itemencoded2item={i:x for i, x in enumerate(items)}

num_users=len(user2user_encoded)
num_items=len(item2item_encoded)

min_rating=min(data['item_rating'])
max_rating=max(data['item_rating'])

data=data.sample(frac=1,random_state=42) #randomly sample the dataset
x=data[['user','item']]

#normalize the targets between 0 and 1 makes it easy to train
y=data['item_rating'].apply(lambda x: (x-min_rating)/(max_rating-min_rating)).values

#assuming training on 90%data and validating on 10%

train_indices=int(0.9*data.shape[0])
x_train,x_val,y_train,y_test = (x[:train_indices],x[train_indices:],y[:train_indices],y[train_indices:])

embedding_size=50

user_ips=layers.Input(shape=[1])
user_embedding=layers.Embedding(num_users,embedding_size,embeddings_initializer='he_normal',embeddings_regularizer=keras.regularizers.l2(1e-6))(user_ips)

user_vect=layers.Flatten()(user_embedding)

item_ips=layers.Input(shape=[1])
item_embedding=layers.Embedding(num_items,embedding_size,embeddings_initializer='he_normal',embeddings_regularizer=keras.regularizers.l2(1e-6))(item_ips)

item_vect=layers.Flatten()(item_embedding)

prod=layers.dot(inputs=[user_vect,item_vect],axes=1)

dense1=layers.Dense(150,activation='relu', kernel_initializer='he_normal')(prod)
dense2=layers.Dense(50,activation='relu',kernel_initializer='he_normal')(dense1)
dense3=layers.Dense(1,activation='relu')(dense2)

model=Model([user_ips,item_ips],dense3)
model.compile(optimizer='adam',loss='mean_squared_error')

history=model.fit([x_train.iloc[:,0], x_train.iloc[:,1]], y_train, batch_size=64, epochs=20, verbose=1)

#pred= model.predict([x_train.iloc[4:5,0], x_train.iloc[4:5,1]])
#pred



st.title('NCF Food Recommender System')

selected_item_name=st.selectbox(
'Your name...',user['user_name'].values)

if st.button('Recommend'):
    customer = selected_item_name
    items_prefered_by_user = data[data.user_name == customer]
    items_not_prefered = item[~item['item_id'].isin(items_prefered_by_user.item.values)]['item_id']

    items_not_prefered_index = [[item2item_encoded.get(x)] for x in items_not_prefered]

    user_encoder = user2user_encoded.get(customer)
    # user_encoder

    user_item_array = np.hstack(([[user_encoder]] * len(items_not_prefered), items_not_prefered_index))

    ratings = model.predict([user_item_array[:, 0], user_item_array[:, 1]]).flatten()

    top_rating_indices = ratings.argsort()[-5:][::-1]

    recommended_item_ids = [itemencoded2item.get(items_not_prefered_index[x][0]) for x in top_rating_indices]

    print('----' * 8)
    print('Top 10 food recommendations')
    print('----' * 8)
    recommended_items = item[item['item_id'].isin(recommended_item_ids)]
    lis = []
    pictures = []
    category=[]
    #if st.button('Recommend'):
    for row in recommended_items.itertuples():
        lis.append(row.item_name)
        pictures.append(img[img['item_name'] == row.item_name]['images'].values[0])
        category.append(row.category)
        #number = 0
    #if st.button('Recommend'):
    #names, pictures = recommend(selected_item_name)


    number = 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(lis[0])
        st.image(pictures[0])
        st.text(category[0])

    with col2:
        st.text(lis[1])
        st.image(pictures[1])
        st.text(category[1])

    with col3:
        st.text(lis[2])
        st.image(pictures[2])
        st.text(category[2])

    with col4:
        st.text(lis[3])
        st.image(pictures[3])
        st.text(category[3])

    with col5:
        st.text(lis[4])
        st.image(pictures[4])
        st.text(category[4])
