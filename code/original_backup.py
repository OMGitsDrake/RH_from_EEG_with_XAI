import time
import numpy as np
import pandas as pd
from glob import glob
# import mne
# from mne.decoding import CSP
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# New import rule from tensorflow
from tensorflow.keras import backend as K

# Which true positive values are predicted as such
# recall = TP/(TP + FN)
'''
    y_true e y_pred are TENSORS
'''
def recall_m(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # TODO why elementwise product of tensors?
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon()) # stochastic error?
    return recall

# Positive preditcions are really positive
# precision = TP/(TP + FP)
'''
    y_true e y_pred are TENSORS
'''
def precision_m(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

'''
    y_true e y_pred are TENSORS
'''
def f1_m(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# base_path = 'C:/Users/alfeo/OneDrive - University of Pisa/1. RESEARCH/1 - XAI for BCI/6 - HR from EEG/code/'
base_path = 'C:/Users/Utente/Drake/UniPi/Tesi/project/'
# "timestamp length"
conditions_ts_len = ['150']  # ['150', '200', '250']
conditions_overlap = ['25', '50', '100', '125']

for i in range(1, 27):
    for cts in conditions_ts_len:
        for cto in conditions_overlap:
            # Get the filename for the dataset
            file_name = glob(base_path + 'data/subject_' + str(i) + '_MADtsROLLINGrawCLASSIFICATION_DATA__'
                             + cts + '_' + cto + '*.csv')[0]    # -> primo file dell'elenco
                                                                # (suppongo ce ne sia 1 solo anyway)

            X = pd.read_csv(file_name).values

            # Normalizzazione e split (?) TODO
            # Normalizes data so that they have mean = 0 and stddev = 1
            # z = (x - u) / s
            # x -> value of a feature
            # u -> mean of that feature
            # s -> stddev of that feature
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Dimensioni dell'input e reshape dei dati per la CNN 1D, considera che la prima meta' dei samples sono
            # peak e il resto no
            n_series = int(file_name[-6:-4])  # Numero di serie temporali per gruppo
            n_samples = int(file_name[-11:-7])  # Numero di gruppi
            max_correlated_channels = 23  # int(file_name[-14:-12])  # Massimo numero di canali correlati
            if cto == '100' or cto == '125':
                overlap = int(file_name[-18:-15])  # Overlap
                n_points = int(file_name[-22:-19]) + 1  # Numero di punti temporali per serie
            else:
                overlap = int(file_name[-17:-15])  # Overlap
                n_points = int(file_name[-21:-18]) + 1  # Numero di punti temporali per serie

            X = X.reshape(n_samples, n_points, n_series)

            file_name = glob(base_path + 'data/subject_' + str(i) + '_MADtsROLLINGrawCLASSIFICATION_LABELS__' +
                             cts + '_' + cto + '*.csv')[0]
            y = pd.read_csv(file_name).values

            #====================MASK====================#

            test_idx_end = int((n_samples // 2) + int(n_samples * 0.1))
            test_idx_begin = int((n_samples // 2) - int(n_samples * 0.1))

            # Create a boolean mask to select elements to keep for training and testing sets
            mask_train = np.ones(X.shape[0], dtype=bool)
            mask_train[test_idx_begin:test_idx_end] = False # da begin a (end - 1) setta False
            X_train = X[mask_train, :, :]
            y_train = y[mask_train]

            mask_test = np.zeros(X.shape[0], dtype=bool)
            mask_test[test_idx_begin:test_idx_end] = True
            X_test = X[mask_test, :, :]
            y_test = y[mask_test]

            # Costruzione del modello
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_points, n_series)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam',
                          metrics=['accuracy', f1_m, precision_m, recall_m])

            # Training e testing
            startTime = time.time()
            model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)
            stopTime = time.time()
            elapsedTime = stopTime-startTime

            loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)

            res = 'Subject, ' + str(i) + ', ts_len: ' + str(n_points) + ', overlap: ' + str(
                overlap) + ', accuracy: ' + str(accuracy) + ', f1_score: ' + str(f1_score) + ', precision: ' + \
                  str(precision) + ', recall: ' + str(recall) + ', elapsed time: ' + str(elapsedTime) + '\n'

            print(res)

            with open(base_path + "results/performanceTS_ROLLINGS_rawCLASSIFICATION_%s.txt" % (time.strftime("%Y%m%d")),
                      "a",
                      encoding="utf-8") as file_object:
                file_object.write(res)
                file_object.close()
