"""Success_hms-v3_KLD-TF.ipynb

# HMS - Harmful Brain Activity Classification

### IMPORTING THE NECESSARY LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')

"""## LOADING THE DATASET"""

BASE_DIR = "/home/m1/23CS60R76/MTP_Works/HMS_Brain_Activity/Harmful_Brain_Activity/"

df = pd.read_csv(f"{BASE_DIR}train.csv")
df.head()

votes_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
df_votes = df[votes_columns].melt(var_name='Brain Activity', value_name='Votes')
df['Brain Activity'] = df[votes_columns].idxmax(axis=1).apply(lambda x: x.replace('_vote', ''))

EEGid_label_list = df[["eeg_id", "Brain Activity", "eeg_label_offset_seconds"]].values.tolist()
EEGid_label_list

"""## MODEL CREATION"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def data_generator(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def createDataset(EEGid_label_list):
    X = []
    y = []
    prev_eegId = ""
    brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
    activity_mapping = {activity.lower(): idx for idx, activity in enumerate(brain_activities)}
    for x in EEGid_label_list:
        eeg_id, label, offset = x
        if(eeg_id!=prev_eegId):
            temp_df = pd.read_parquet(f'{BASE_DIR}train_eegs/{eeg_id}.parquet')
            # C1 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
            # C2 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
            # temp_arr1 = temp_df[C1].to_numpy()
            # temp_arr1[np.isnan(temp_arr1)] = 0
            # temp_arr2 = temp_df[C2].to_numpy()
            # temp_arr2[np.isnan(temp_arr2)] = 0
            # temp_arr = temp_arr1 - temp_arr2
            C = ['Fp1', 'Fp2', 'F7', 'T3', 'P4', 'C4']
            temp_arr = temp_df[C].to_numpy()
            temp_arr[np.isnan(temp_arr)] = 0
        start = 200*int(offset)
        X.append(temp_arr[start:start+1000])
        y.append(activity_mapping[label])
        # print(eeg_id, activity_mapping[label], temp_arr.shape)
        prev_eegId = eeg_id
    X = np.array(X)
    y = np.array(y)
    print("X:", X.shape)
    print("y:", y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test, random_state=42)
    return (X_train, X_val, X_test, y_train, y_val, y_test)

## deep learning model
n_classes = 6

def createModel(ip_shape):
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
    
        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0,):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs)
    
    model = build_model(ip_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25,)
    LEARNING_RATE = 1e-5
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

X_train, X_val, X_test, y_train, y_val, y_test = createDataset(EEGid_label_list)
model = createModel(X_train.shape[1:])
model.summary()

# callbacks
VERBOSE=0
#lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=VERBOSE, min_le=1e-8)
es = EarlyStopping(monitor='val_loss', patience=50, verbose=VERBOSE, mode='auto', restore_best_weights=True)
checkpoint_filepath = '/home/m1/23CS60R76/MTP_Works/BestSavedModels/Trans_HMS_model_v1.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
callbacks = [es, model_checkpoint_callback]
hist = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_val,y_val),
    epochs=250,
    batch_size=16,
    callbacks=callbacks
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Saving the figure.
plt.savefig("trans_training_history-0.jpg")