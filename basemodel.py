import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import shutil

from sklearn.model_selection import train_test_split

from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import TimeDistributed
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.models import load_model


wlasl_df = pd.read_json("WLASL_v0.3.json")

#check if the video id is available in the dataset
#and return the viedos ids of the current instance

def get_videos_ids(json_list):
    video_ids = []
    for ins in json_list:
        video_id = ins['video_id']
        if os.path.exists(f'videos/{video_id}.mp4'):
            video_ids.append(video_id)
    return video_ids

#function to check if the video id is available in the dataset
#and return the viedos ids and url or any other featrue of the current instance

def get_json_features(json_list):

    videos_ids = []
    videos_urls = []
    videos_bbox = []
    videos_fps = []
    videos_frame_end = []
    videos_frame_start = []
    videos_signer_id = []
    videos_source = []
    videos_split = []
    videos_variation_id = []
    for ins in json_list:

        video_id = ins['video_id']
        video_url = ins['url']
        video_bbox = ins['bbox']
        video_fps = ins['fps']
        video_frame_end = ins['frame_end']
        video_frame_start = ins['frame_start']
        video_signer_id = ins['signer_id']
        video_source = ins['source']
        video_split = ins['split']
        video_variation_id = ins['variation_id']
        if os.path.exists(f'videos/{video_id}.mp4'):
            videos_ids.append(video_id)
            videos_urls.append(video_url)
            videos_bbox.append(video_bbox)
            videos_fps.append(video_fps)
            videos_frame_end.append(video_frame_end)
            videos_frame_start.append(video_frame_start)
            videos_signer_id.append(video_signer_id)
            videos_source.append(video_source)
            videos_split.append(video_split)
            videos_variation_id.append(video_variation_id)
    return videos_ids, videos_urls, videos_bbox, videos_fps, videos_frame_end, videos_frame_start, videos_signer_id, videos_source, videos_split,videos_variation_id

#extract video_id
wlasl_df["video_ids"] = wlasl_df["instances"].apply(get_videos_ids)

# extract each feature into a column of df
features_df = pd.DataFrame(columns=['gloss', 'video_id', 'urls', 'bbox', 'fps', 'frame_end', 'frame_start','signer_id', 'source', 'split', 'variation_id'])
for row in wlasl_df.iterrows():
    ids, urls, bbox, fps, frame_end, frame_start,signer_id, source, split, variation_id = get_json_features(row[1][1])
    word = [row[1][0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids, urls, bbox, fps, frame_end, frame_start, signer_id, source, split, variation_id)), columns=features_df.columns)
    features_df = features_df.append(df, ignore_index=True)

# take a subset of the full data using nslt_ file

# Load the JSON data from the file
with open('nslt_100.json', 'r') as file:
    nslt_100_data = json.load(file)

# Create a list of tuples for each record
data_100 = [(key, value['subset'], value['action']) for key, value in nslt_100_data.items()]

# Create a DataFrame from the data
df_100 = pd.DataFrame(data_100, columns=['ID', 'Subset', 'Action'])

# Display the first few rows of the DataFrame
print(df_100.head())


merged_df = pd.merge(df_100, features_df, left_on='ID', right_on='video_id')

# Display the merged DataFrame
merged_df


# split data into test train validation
train_mask = merged_df['split'] == 'train'
val_mask = merged_df['split'] == 'val'
test_mask = merged_df['split'] == 'test'

train_pos = np.flatnonzero(train_mask)
val_pos =  np.flatnonzero(val_mask)
test_pos = np.flatnonzero(test_mask)

train = merged_df.iloc[train_pos]

val = merged_df.iloc[val_pos]

test = merged_df.iloc[test_pos]

# store splited data into different folder
from distutils.dir_util import copy_tree

output_folder = 'working/'

def generateDatasplitFolder(series, folderName):
    new_path = output_folder+'data/'+str(folderName)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        for val in series:
            from_directory = 'videos/'+str(val)+'.mp4'
            to_directory = new_path+str(val)+'.mp4'
            shutil.copy(from_directory, to_directory)

# train data
generateDatasplitFolder(train['video_id'], 'train/')

# validation data
generateDatasplitFolder(val['video_id'], 'val/')

# test data
generateDatasplitFolder(test['video_id'], 'test/')

import random
import math

# Constants
IMG_SIZE = 256  #frame size after resizing
SEQUENCE_LENGTH = 20
CROP_SIZE = 224


def resize_frame(frame, min_size=256):
    h, w = frame.shape[:2]

    # Upscale if the frame is smaller than 226 in either dimension
    if w < 226 or h < 226:
        d = 226. - min(w, h)
        scale_factor = 1 + d / min(w, h)
        frame = cv2.resize(frame, dsize=(0, 0), fx=scale_factor, fy=scale_factor)

    # Downscale if the frame is larger than 256 in either dimension
    if w > 256 or h > 256:
        frame = cv2.resize(frame, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

    return frame

def normalize_frame(frame):
    frame = frame / 255.0  # Convert pixel values to range [0, 1]
    return frame

def crop_frame(frame, crop_size=224):
    h, w = frame.shape[:2]
    if h == crop_size and w == crop_size:
        return frame  # Return the frame as-is since it's already the right size

    # If either dimension is larger than crop_size, perform the crop
    if h > crop_size:
        start_y = random.randint(0, h - crop_size)
    else:
        start_y = 0  # Can't crop in the y-dimension, so start at 0

    if w > crop_size:
        start_x = random.randint(0, w - crop_size)
    else:
        start_x = 0  # Can't crop in the x-dimension, so start at 0

    cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cropped

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = resize_frame(frame)
            frame = normalize_frame(frame)
            frame = crop_frame(frame)
            frames.append(frame)
    finally:
        cap.release()

    # Frame selection and transformation
    frames = frames[:SEQUENCE_LENGTH]  # Select the first SEQUENCE_LENGTH frames

    # Data padding
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])  # Repeat the last frame

    return np.array(frames)


label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train['gloss'])
)
print(label_processor.get_vocabulary())

def prepare_videos_and_labels(df, root_dir):
    x = []  # List to hold video data
    y = []  # List to hold labels

    for i, (video_id, gloss) in df[['video_id', 'gloss']].iterrows():
        video_path = f'working/data/{root_dir}/{video_id}.mp4'
        frames = preprocess_video(video_path)
        label = label_processor([gloss]).numpy()[0]

        x.append(frames)
        y.append(label)

    return x, y

# Preparing data and labels
x_train, y_train = prepare_videos_and_labels(train, 'train')
x_test, y_test = prepare_videos_and_labels(test, 'test')
x_val, y_val = prepare_videos_and_labels(val, 'val')


x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print("Shape of x_train: ", x_train.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of x_test: ", x_test.shape)

print("Shape of y_train: ", y_train.shape)
print("Shape of y_val: ", y_val.shape)
print("Shape of y_test: ", y_test.shape)


n_labels = 100 #Number of classes
Learning_rate = 0.0001
batch_size = 30
num_epochs = 50
img_channel = 3 #RGB Channels

#Implement CNN-LSTM Model

#Input Layer for CNN-LSTM Model
video = Input(shape=(SEQUENCE_LENGTH,CROP_SIZE,CROP_SIZE,img_channel))  # 64, 224, 224, 3
#Load transfer learning model, MobileNet + ImageNet weights
model = applications.MobileNet(input_shape=(CROP_SIZE,CROP_SIZE,img_channel),
                               weights="imagenet", include_top=False)
model.trainable = False
# Fully Connected Dense Layer
inputs = model.output
x = Flatten()(inputs)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.3)(x)
cnn_out = Dense(128, activation="relu")(x)

#CNN Layers
Lstm_inp = Model(inputs=model.input, outputs=cnn_out)
#Distribute CNN output by timesteps,
encoded_frames = TimeDistributed(Lstm_inp)(video)

#LSTM Layers
encoded_sequence = LSTM(256)(encoded_frames)
hidden_dropout = Dropout(0.3)(encoded_sequence)
hidden_layer = Dense(128, activation="relu")(encoded_sequence)
outputs = Dense(n_labels, activation="softmax")(hidden_layer)

# Contruct CNN-LSTM model
model = Model([video], outputs)

adam_opt = keras.optimizers.Adam(learning_rate=Learning_rate)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=adam_opt,
              metrics=["accuracy"])

#Train Model
hist = model.fit(x_train, y_train, batch_size=batch_size,
                 validation_data=(x_val, y_val), shuffle=True,
                 epochs=num_epochs)

import pickle

# Serialize the history object
with open('history.pkl', 'wb') as f:
    pickle.dump(hist.history, f)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Extracting data from the history object
history_dict = hist.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# Plotting
plt.figure(figsize=(12, 5))

# Plot for standard accuracy
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# Plot for loss
plt.subplot(1, 2, 2)  # 1 row, 3 columns, 3rd subplot
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_validation_accuracy_loss.png')
plt.close()