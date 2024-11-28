import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
import streamlit as st
import time
import io
import zipfile
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

def create_zip(strings, filenames):
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for string, filename in zip(strings, filenames):
            zip_file.writestr(filename, string)
    
    zip_buffer.seek(0)
    return zip_buffer

def create_image_zip(images, filenames):
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for image, filename in zip(images, filenames):
            zip_file.writestr(filename, image.read())
            
    
    zip_buffer.seek(0)
    return zip_buffer
#==============Data structures==============#
# ES = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     min_delta=0.1,
#     mode='min',
#     restore_best_weights=True
# )

if "png_buffers" not in st.session_state:
    st.session_state.png_buffers = []

if "txt_buffers" not in st.session_state:
    st.session_state.txt_buffers = []

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
st.title("Retrieve HR from EEG using a Convolutional Neural Network")
st.markdown('''
    Please upload files **which file name formats as follows**:
    subject\_*[i]*\_MADtsROLLINGrawCLASSIFICATION\_[DATA/LABELS]\_\_*[wlen]*\_*[overlap]*\_BK\_*[grps]*\_*[series]*.csv\n
    Where:\n
    - **i** is the subject taken in exam
    - **wlen** is the length of each time window
    - **overlap** is the amount of data overlap between each time window
    - **grps** is the number of groups in the time series
    - **series** is the number of correlated channels
''')

overlap = st.number_input("Select the time window overlap", 25, 125, "min", 5, "%d")
n_points = st.number_input("Select the number of points per time window", 150, 250, "min", 50, "%d") + 1
window_printed = st.number_input("Select the number of windows extracted per subject", 1, 10, "min", 1, "%d")

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

totalTime = 0

if (len(data) == len(labels)) and data and labels and s:
    scaler = StandardScaler()
    with st.spinner('Working...'):
        n_samples = int(data_files[0].name[-11:-7])
        for i in range(n_soggetti):
            if n_soggetti < 2:
                LOG("Loading data and labels")
                
                multiple_subj = False

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

                LOG("Data and labels loaded!")
                LOG(f'Train data: {X_train.shape}, test data: {X_test.shape}, train labels: {y_train.shape} test labels: {y_test.shape}')
            else:
                #---------------------------DATA GENERATION-------------------------------#
                if n_soggetti > 1:
                    n_samples_overall = 0
                    X_train = []
                    X_test = []
                    LOG(f"Loading data of subject {i+1}...")
                    for j in range(n_soggetti):
                        # Dataset di test
                        if j == i:
                            X_test = data[i].values
                            X_test = scaler.fit_transform(X_test)
                            X_test = X_test.reshape(n_samples, n_points, n_series)
                            continue

                        # Dataset di training
                        X_train.extend(data[j].values)  # append to the whole time series
                        n_samples_overall += int(data_files[j].name[-11:-7])

                    X_train = scaler.fit_transform(X_train)
                    X_train = X_train.reshape(n_samples_overall, n_points, n_series)
                
                LOG("Data loaded!")
                LOG(f'Train data: {X_train.shape}')
                LOG(f'Test data: {X_test.shape}')

                #---------------------------LABEL GENERATION-------------------------------#

                y_train = []
                y_test = []
                first_iter = True
                LOG(f"Loading labels of subject {i+1}...")

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

                LOG("Labels loaded!")
                LOG(f'Training labels: {y_train.shape}')
                LOG(f'Testing labels: {y_test.shape}')
            
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

            startTime = time.time()
            # Training
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
            plt.plot(loss, label='Training', color='cyan')
            plt.plot(val_loss, label='Validation', color='violet')
            plt.xlabel('Epochs')
            plt.title('Loss')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            st.pyplot(plt)

            # Testing
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
    
            res = f''':blue[Accuracy]: {cnn_stats["acc"][i]:.4f}\n
:orange[F1 score]: {cnn_stats["f1"][i]:.4f}\n
:green[Precision]: {cnn_stats["precision"][i]:.4f}\n
:red[Recall]: {cnn_stats["recall"][i]:.4f}\n
:violet[Loss]: {cnn_stats["loss"][i]:.4f}
'''

            endTime = time.time()
            totalTime += endTime-startTime
            
            st.divider()
            st.header(f'RESULTS:')
            st.markdown(res)

            if len(st.session_state.txt_buffers) <= i:
                st.session_state.txt_buffers.append(f'{i+1}) {time.strftime("%d-%m @ %H:%M")}\n' + res)

            r = np.random.randint(0, len(y_test))
            LOG(f'Starting label: {r}')
            
            start = n_points * r
            end = n_points * (r + window_printed)
            t_window = X_test.flatten()[start:end]
            
            #----DEBUG----
            plt.figure(figsize=(12, 5))
            plt.plot(X_test.flatten(), linewidth=1)
            plt.grid(alpha=0.3)
            plt.title('Original time series')
            st.pyplot(plt)
            plt.figure(figsize=(12, 5))
            plt.plot(t_window, linewidth=1)
            plt.grid(alpha=0.3)
            plt.title('Original time windows')
            st.pyplot(plt)
            #----DEBUG----

            plt.figure(figsize=(12, 5))
            for k in range(window_printed):
                to_print = t_window[k*n_points:(k+1)*n_points]
                
                if y_test[k]:
                    plt.scatter(np.argmax(to_print)+(k*n_points), np.max(to_print), color='blue', marker='s', alpha=0.5)
                color = 'green' if y_pred[k] and y_test[k] else 'red'
                plt.plot(range(n_points*k, n_points*(k+1)), to_print, color=color)

                plt.axvline(x=(k*n_points), color='orange', linestyle='--', linewidth=1, alpha=0.6)
                plt.axvline(x=len(to_print)*(k+1)-overlap, color='violet', linestyle='--', linewidth=1, alpha=0.8)
                plt.axvline(x=overlap+k*n_points, color='violet', linestyle='--', linewidth=1, alpha=0.8)
            plt.axhline(y=0, color='black', alpha=0.4, linewidth=1)
            plt.xlim([0,len(t_window)])
            plt.title(f'Time Windows starting from {r}-th')
            plt.grid(alpha=0.3)

            st.pyplot(plt)

            st.subheader('Legend')
            st.markdown('''
            **:green[- Predicted well]**\n
            **:red[- Mistaken]**\n
            **:blue[- Peak value (if label present)]**\n
            **:violet[- Overlap delimiter]**\n
            **:orange[- Window delimiter]**\n
            ''')

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)

            if len(st.session_state.png_buffers) <= window_printed:
                st.session_state.png_buffers.append(img_buf)

    st.divider()

    if multiple_subj:
        st.write(f'Best Accuracy score: {cnn_best_worst_acc_case[0]:.6f} with subject {cnn_best_worst_acc_case[1]}')
        st.write(f'Worst Accuracy score: {cnn_best_worst_acc_case[2]:.6f} with subject {cnn_best_worst_acc_case[3]}')
    
        st.write(f'Avg computation time: {totalTime/n_soggetti:.2f}')

        sub_label = [f'Subject {j+1}' for j in range(n_soggetti)]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(cnn_stats['acc'], label='Accuracy', marker='.', linestyle='-')
        plt.plot(cnn_stats['f1'], label='F1 score', marker='.', linestyle='-')
        plt.plot(cnn_stats['precision'], label='Precision', marker='.', linestyle='-')
        plt.plot(cnn_stats['recall'], label='Recall', marker='.', linestyle='-')
        plt.xlabel('Subjects')
        plt.ylabel('Metrics')
        plt.xticks(np.array(range(n_soggetti)), sub_label, rotation=45)
        plt.title('CNN Metrics')
        plt.grid(alpha=0.5)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(cnn_stats['loss'], label='Loss', marker='.', linestyle='-')
        plt.xlabel('Subjects')
        plt.ylabel('Loss')
        plt.xticks(np.array(range(n_soggetti)), sub_label, rotation=45)
        plt.title('CNN Loss')
        plt.grid(alpha=0.5)
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

    filenames = [f"Subject_{i+1}.txt" for i in range(n_soggetti)]
    txt_zip = create_zip(st.session_state.txt_buffers, filenames)
    st.download_button(
        label="Download Results",
        data = txt_zip,
        file_name='archivio.zip',
        mime = "application/zip",
        key = 'Text'
    )

    filenames = [f"img_{i+1}.png" for i in range(n_soggetti*window_printed)]
    png_zip = create_image_zip(st.session_state.png_buffers, filenames)
    st.download_button(
        label="Download Images",
        data = png_zip,
        file_name= "archivio_immagini.zip",
        mime = "application/archive",
        key = 'Images'
    )