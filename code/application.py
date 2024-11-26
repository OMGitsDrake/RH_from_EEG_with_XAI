import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import StandardScaler
# New import rule from tensorflow
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Disable all warnings
warnings.filterwarnings("ignore")

def recall_m(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def LOG(msg, lvl='INFO'):
    print(f"{lvl}: {str(msg)}")

#==============Data structures==============#
# ES = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     min_delta=0.1,
#     mode='min',
#     restore_best_weights=True
# )

cnn_stats = {
    'acc':[],
    'f1':[],
    'precision':[],
    'recall':[],
    'loss':[]
}

# best, id, worst, id
cnn_best_worst_acc_case = [0, -1, np.inf, -1]

base_path = "C:/Users/Utente/Drake/UniPi/Tesi/RH_from_EEG_with_XAI/"

#==============UI display==============#
st.title("Retrieve HR from EEG")
st.markdown('''
    Please upload files **which file name formats as follows (for the data)**:
    subject\_*i*\_MADtsROLLINGrawCLASSIFICATION\_DATA\_\_*ts\_len*\_*overlap*\_BK\_*n\_groups*\_*n\_series*.csv\n
    Where:\n
    - **i** is the subject taken in exam
    - **ts_len** is the length of the time series
    - **overlap** is the amount of data overlap between each time series
    - **n_group** is the number of groups in the time series
    - **n_series** is the number of time series per group
''')

overlap = st.number_input("Select the time window overlap", 25, 125, "min", 5, "%d")
n_points = st.number_input("Select the number of points per time window", 150, 250, "min", 50, "%d") + 1

data = []
data_files = st.file_uploader("Upload the EEG file for the data", type="csv", accept_multiple_files=True, help='Upload CSV files representing the EEG of the subjects')
for f in data_files:
    data.append(pd.read_csv(f))

labels = []
label_files = st.file_uploader("Upload the labels", type="csv", accept_multiple_files=True, help='Upload CSV files representing the labels for the subjects')
for f in label_files:
    labels.append(pd.read_csv(f))

s = st.button('Start computation')

n_soggetti = len(data)
n_series = 23  # Numero di serie temporali per window

multiple_subj = True

if (len(data) == len(labels)) and data and labels and s:
    scaler = StandardScaler()
    with st.spinner('Working...'):
        for i in range(n_soggetti):
            if n_soggetti < 2:
                multiple_subj = False

                n_samples = int(data_files[0].name[-11:-7])
                X = data[0].values.reshape(n_samples, n_points, n_series)
                y = labels[0].values

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

                X_train_2D = X_train.reshape(X_train.shape[0], -1)
                X_test_2D = X_test.reshape(X_test.shape[0], -1)

                # Apply StandardScaler
                X_train = scaler.fit_transform(X_train_2D)
                X_test = scaler.transform(X_test_2D)

                # Reshape back to 3D for CNN
                X_train = X_train.reshape(X_train.shape[0], n_points, n_series)
                X_test = X_test.reshape(X_test.shape[0], n_points, n_series)

                st.write("Data and labels loaded!")
                LOG(f'Data train: {X_train.shape}, data test: {X_test.shape}, train labels: {y_train.shape} test labels: {y_test.shape}')
            else:
                #---------------------------DATA GENERATION-------------------------------#
                if n_soggetti > 1:
                    n_samples = 0
                    X_train = []
                    X_test = []
                    st.write(f"Loading data of subject {i+1}...")
                    for j in range(n_soggetti):
                        # Dataset di test
                        if j == i:
                            X_test = data[i].values
                            X_test = scaler.fit_transform(X_test)
                            n_samples_i = int(data_files[i].name[-11:-7])
                            X_test = X_test.reshape(n_samples_i, n_points, n_series)
                            continue

                        # Dataset di training
                        X_train.extend(data[j].values)  # append to the whole time series
                        n_samples += int(data_files[j].name[-11:-7])

                    X_train = scaler.fit_transform(X_train)
                    X_train = X_train.reshape(n_samples, n_points, n_series)
                
                st.write("Data loaded!")
                LOG(X_train.shape)
                LOG(X_test.shape)

                #---------------------------LABEL GENERATION-------------------------------#

                y_train = []
                y_test = []
                first_iter = True
                st.write(f"Loading labels of subject {i+1}...")

                for j in range(n_soggetti):
                    # Label di testing
                    if j == i:
                        y_test = labels[i].values
                        continue
                    # Label di training
                    if first_iter:
                        y_train = labels[j].values
                        first_iter = False
                    else:
                        y_train = np.append(y_train, labels[j].values, axis=0)
                        # y_train.extend(labels[j].values)

                st.write("Labels loaded!")
                LOG(y_train.shape)
                LOG(y_test.shape)
            
            #------------------------CNN------------------------#
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_points, n_series)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            # model.add(Dense(1024, activation='relu'))
            # model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam',
                            metrics=['accuracy', f1_m, precision_m, recall_m])

            # Training
            st.write("Training...")
            history = model.fit(X_train, y_train,
                                epochs=100, batch_size=64,
                                validation_data=(X_test, y_test),
                                verbose=0)

            accuracy = history.history['accuracy']
            val_accuracy = history.history['val_accuracy']

            f1 = history.history['f1_m']
            val_f1 = history.history['val_f1_m']
            
            precision = history.history['precision_m']
            val_precision = history.history['val_precision_m']
            
            recall = history.history['recall_m']
            val_recall = history.history['val_recall_m']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            st.header(f'Subject {i+1}:')
            
            # training
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 3, 1)
            plt.plot(accuracy, label='Accuracy')
            plt.plot(f1, label='f1 score')
            plt.plot(precision, label='Precision')
            plt.plot(recall, label='Recall')
            plt.xlabel('Epochs')
            plt.title('Training Metrics')
            plt.grid(True)
            plt.legend()

            # validation
            plt.subplot(1, 3, 2)
            plt.plot(val_accuracy, label='Accuracy')
            plt.plot(val_f1, label='f1 score')
            plt.plot(val_precision, label='Precision')
            plt.plot(val_recall, label='Recall')
            plt.xlabel('Epochs')
            plt.title('Validation Metrics')
            plt.grid(True)
            plt.legend()

            # loss
            plt.subplot(1, 3, 3)
            plt.plot(loss, label='Training')
            plt.plot(val_loss, label='Validation')
            plt.xlabel('Epochs')
            plt.title('Loss')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            st.pyplot(plt)

            # Testing
            st.write("Testing...")
            loss, accuracy, f1, precision, recall = model.evaluate(X_test, y_test, verbose=0)

            if multiple_subj:
                if accuracy > cnn_best_worst_acc_case[0]:
                    cnn_best_worst_acc_case[0] = accuracy
                    cnn_best_worst_acc_case[1] = i+1
                if accuracy < cnn_best_worst_acc_case[2]:
                    cnn_best_worst_acc_case[2] = accuracy
                    cnn_best_worst_acc_case[3] = i+1

            cnn_stats['acc'].append(accuracy)
            cnn_stats['f1'].append(f1)
            cnn_stats['loss'].append(loss)
            cnn_stats['precision'].append(precision)
            cnn_stats['recall'].append(recall)

            # model predictions
            predictions = model.predict(X_test)
            y_pred = (predictions > 0.5).astype(int)
    
            res = f'Accuracy:\t{cnn_stats["acc"][i]:.4f}\n\
F1 score:\t{cnn_stats["f1"][i]:.4f}\n\
Precision:\t{cnn_stats["precision"][i]:.4f}\n\
Recall:\t{cnn_stats["recall"][i]:.4f}\n\
Loss:\t{cnn_stats["loss"][i]:.4f}\n\n'

            st.header(f'RESULTS:')
            st.write(res)

            # Rimpiazzare con download file.txt
            with open(base_path + "results/performanceTS_ROLLINGS_rawCLASSIFICATION_%s.txt" % (time.strftime("%Y%m%d")),
                      "a",
                      encoding="utf-8") as file_object:
                file_object.write(f'{i+1}) {time.strftime("%d%H%M")}\n' + res)
                file_object.close()

            # Plottare t_window in modo da avere una window randomica da far vedere anche dove e' posizionata la labl in caso di classe positiva
            for i in range(5):
                # series = X_test.reshape(-1)[0:n_points]
                # r = next((i for i, x in enumerate(y_test) if x == 0), -1)
                r = np.random.randint(0, len(y_test))
                LOG(f'Index of label: {r}')
                t_window = X_test.reshape(-1)[n_points*r:n_points*(r+1)]
                m = np.argmax(t_window)
                LOG(f't_window argmax: {m}')
                plt.figure(figsize=(12, 5))
                plt.plot(t_window, alpha=0.7)
                plt.plot(0, color='black', alpha=0.5)
                # il primo parametro e' difficile da trovare
                plt.scatter(len(t_window)//2, y_test[r], color='green', marker='+', alpha=0.7, label='Label')
                plt.scatter(len(t_window)//2, y_pred[r], color='red', marker='x', alpha=0.5, label='Prediction')
                plt.scatter(m, np.max(t_window), color='blue', marker='s', alpha=0.7, label='Peak value')
                plt.axvline(x=(len(t_window)-overlap), color='red', linestyle='--', linewidth=1, alpha=0.6)
                plt.axvline(x=overlap, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Overlap delimiter')
                plt.xlim([0,len(t_window)])
                plt.title(f'Time Window #{r}')
                plt.grid(alpha=0.3)
                plt.legend()

                st.pyplot(plt)

            # os.makedirs(os.path.join(data_path, 'figures'), exist_ok=True)
            # plt.savefig(data_path + f'figures/reg_{i}_{cts}_{cto}.png')

            # ind = np.arange(len(y_pred))
            # plt.scatter(ind, y_test, color='green', marker='.', alpha=0.7, label='Labels')
    if multiple_subj:
        st.write(f'Best Accuracy score: {cnn_best_worst_acc_case[0]:.6f} with subject {cnn_best_worst_acc_case[1]}')
        st.write(f'Worst Accuracy score: {cnn_best_worst_acc_case[2]:.6f} with subject {cnn_best_worst_acc_case[3]}')

    # sub_label = [f'Subject {j+1}' for j in range(n_soggetti)]
    # label_pos = [(n_overlap*j + (n_overlap - 1)/2) for j in range(n_soggetti)]

    # plt.figure(figsize=(17, 5))
    # plt.plot(series, cnn_stats['acc'], label='Accuracy', marker='.', linestyle='-')
    # plt.plot(series, cnn_stats['f1'], label='F1 score', marker='.', linestyle='-')
    # plt.plot(series, cnn_stats['precision'], label='Precision', marker='.', linestyle='-')
    # plt.plot(series, cnn_stats['recall'], label='Recall', marker='.', linestyle='-')
    # plt.xlabel('Overlap by subject')
    # plt.ylabel('Metrics')
    # plt.xticks(label_pos, sub_label, rotation=45)
    # plt.title('CNN Metrics over multiple overlap len')
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1, 2, 1)
    # plt.plot(y_, cnn_stats['loss'], label='Loss', marker='.', linestyle='-')
    # plt.xlabel('Overlap by subject')
    # plt.ylabel('Loss')
    # plt.xticks(label_pos, sub_label, rotation=45)
    # plt.title('CNN Loss over multiple overlap len')
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(indices, cnn_stats['loss'], label='Loss', marker='.', linestyle='-')
    # plt.xlabel('Overlap by subject')
    # plt.ylabel('Loss')
    # plt.xticks(label_pos, sub_label, rotation=45)
    # plt.title('CNN Loss over multiple overlap len')
    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()
    # st.pyplot(plt)

            # series = data[i].values.reshape(-1)
            # X = data[i].values
            # file_name = files[i].name

            # # Split e normalizzazione
            # scaler = StandardScaler()

            # #==========CNN SETUP==========#
            # X = X.reshape(n_samples, n_points, n_series)

            # # LABELS
            # y = labels[i].values

            # test_idx_end = int((n_samples // 2) + int(n_samples * 0.1))
            # test_idx_begin = int((n_samples // 2) - int(n_samples * 0.1))

            # # Create a boolean mask to select elements to keep for training and testing sets
            # mask_train = np.ones(X.shape[0], dtype=bool)
            # mask_train[test_idx_begin:test_idx_end] = False # da begin a (end - 1) setta False
            # X_train = X[mask_train, :, :]
            # y_train = y[mask_train]

            # mask_test = np.zeros(X.shape[0], dtype=bool)
            # mask_test[test_idx_begin:test_idx_end] = True
            # X_test = X[mask_test, :, :]
            # y_test = y[mask_test]

            # X_train_2D = X_train.reshape(X_train.shape[0], -1)
            # X_test_2D = X_test.reshape(X_test.shape[0], -1)

            # # Apply StandardScaler
            # X_train = scaler.fit_transform(X_train_2D)
            # X_test = scaler.transform(X_test_2D)

            # # Reshape back to 3D for CNN
            # X_train = X_train.reshape(X_train.shape[0], n_points, n_series)
            # X_test = X_test.reshape(X_test.shape[0], n_points, n_series)
