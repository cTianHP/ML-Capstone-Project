import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os

tourist_attraction = pd.read_excel('C:/Users/ASUS/Downloads/Model V4/Tourist Attraction_List.xlsx')
tourist_attraction.head(10)

tourist_attraction.isna().sum()

tourist_attraction.info()

user_rating = pd.read_excel('C:/Users/ASUS/Downloads/Model V4/Transformed User Rating Tempat Wisata.xlsx')
user_rating.head(10)

user_rating.isna().sum()

user_rating.info()

place_rank = user_rating['place_id'].value_counts().reset_index()
place_rank

df = user_rating.copy()
df.head()

def dict_encoder(col, data=df):

  unique_val = data[col].unique().tolist()

  val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}

  val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
  return val_to_val_encoded, val_encoded_to_val

user_to_user_encoded, user_encoded_to_user = dict_encoder('user_id')

df['user'] = df['user_id'].map(user_to_user_encoded)

place_to_place_encoded, place_encoded_to_place = dict_encoder('place_id')

df['place'] = df['place_id'].map(place_to_place_encoded)

df

num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)

df['rating'] = df['rating'].values.astype(np.float32)

min_rating, max_rating = min(df['rating']), max(df['rating'])

print(f'Number of User: {num_users}, Number of Place: {num_place}, Min Rating: {min_rating}, Max Rating: {max_rating}')

df = df.sample(frac=1, random_state=42)
df.head(2)

x = df[['user', 'place']].values

y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

x_train_array = [x_train[:, 0], x_train[:, 1]]
x_val_array = [x_val[:, 0], x_val[:, 1]]

def RecommenderSystem_Model():
    embedding_size = 1024

    user = tf.keras.layers.Input(shape = [1])
    user_embedding = tf.keras.layers.Embedding(input_dim = num_users, output_dim = embedding_size)(user)

    place = tf.keras.layers.Input(shape = [1])
    place_embedding = tf.keras.layers.Embedding(input_dim = num_place, output_dim = embedding_size)(place)

    x = tf.keras.layers.Dot(axes=2)([user_embedding, place_embedding])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(6, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=[user, place], outputs=x)

    return model

model = RecommenderSystem_Model()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = model.fit(
    x=x_train_array,
    y=y_train,
    batch_size=32,
    epochs=1000,
    verbose=1,
    validation_data=(x_val_array, y_val),
    callbacks=[early_stopping]
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()

model.save('model.h5')

model.save('saved_model', save_format='tf')

place_df = tourist_attraction[['place_id','name','category','rating','latitude','longitude']]
df = user_rating.copy()

place_df

user_id = df.user_id.sample(1).iloc[0]
place_visited_by_user = df[df.user_id == user_id]

place_visited_by_user

place_not_visited = place_df[~place_df['place_id'].isin(place_visited_by_user.place_id.values)]['place_id']
place_not_visited = list(
    set(place_not_visited)
    .intersection(set(place_to_place_encoded.keys()))
)

place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(
    ([[user_encoder]] * len(place_not_visited), place_not_visited)
)
user_place_array = [user_place_array[:, 0], user_place_array[:, 1]]

ratings = model.predict(user_place_array).flatten()
top_ratings_indices = ratings.argsort()[-7:][::-1]
recommended_place_ids = [
    place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
]

print('Daftar rekomendasi untuk: {}'.format('User ' + str(user_id)))
print('===' * 15,'\n')
print('----' * 15)
print('Tempat dengan rating wisata paling tinggi dari user')
print('----' * 15)

top_place_user = (
    place_visited_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .place_id.values
)

place_df_rows = place_df[place_df['place_id'].isin(top_place_user)]
for row in place_df_rows.itertuples():
    print(row.name, ':', row.category)

print('')
print('----' * 15)
print('Top 7 place recommendation')
print('----' * 15)

recommended_place = place_df[place_df['place_id'].isin(recommended_place_ids)]
for row, i in zip(recommended_place.itertuples(), range(1,8)):
    print(i,'.', row.name, '\n    ', row.category, ',', 'Rating Wisata ', row.rating,'\n')

print('==='*15)

recommended_place

bus_route = pd.read_excel('C:/Users/ASUS/Downloads/Model V4/bus_route_merged.xlsx')
bus_route

from geopy.distance import geodesic

def find_nearest_bus_route(place, bus_route):
    min_distance = float('inf')
    nearest_bus_route = None

    for _, bus_route in bus_route.iterrows():
        distance = geodesic((place['latitude'], place['longitude']),
                            (bus_route['latitude'], bus_route['longitude'])).kilometers

        if distance < min_distance:
            min_distance = distance
            nearest_bus_route = bus_route

    return nearest_bus_route, min_distance

result_rows = []

for _, place in recommended_place.iterrows():
    nearest_bus_stop, distance = find_nearest_bus_route(place, bus_route)
    result_rows.append({
        'place_name': place['name'],
        'place_latitude': place['latitude'],
        'place_longitude': place['longitude'],
        'nearest_bus_stop': nearest_bus_stop['nama'],
        'bus_stop_latitude': nearest_bus_stop['latitude'],
        'bus_stop_longitude': nearest_bus_stop['longitude'],
        'distance_to_nearest_bus_stop': distance
    })

result_df = pd.DataFrame(result_rows)

result_df

recommended_tourist_attraction = result_df[['place_name', 'place_latitude','place_longitude']]
recommended_tourist_attraction

recommended_bus_route = result_df[['nearest_bus_stop', 'bus_stop_latitude', 'bus_stop_longitude']]
recommended_bus_route