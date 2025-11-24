import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# CSV'den verileri yükleme
data_path = "C:/Users/Hp/Desktop/python.py/fer2013.csv"  # CSV dosyasının yolu
data = pd.read_csv(data_path)

# Veriyi ve etiketleri ayırma
X = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48, 1)).values
y = data['emotion'].values

# Veriyi normalize etme
X = np.stack(X)
X = X / 255.0

# Etiketleri one-hot encode yapma
y = tf.keras.utils.to_categorical(y, num_classes=7)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN modelinin tanımlanması
model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(48, 48, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, kernel_initializer='he_normal'))  # Duygusal sınıflar
model.add(Activation('softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])


# Modelin eğitilmesi
history = model.fit(X_train, y_train, validation_split=0.3, epochs=50, batch_size=64)

# Eğitim sonucunu görselleştirme
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.ylabel('%')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

# Kayıp (Loss) grafiği
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.ylabel('%')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

# Modelin tahminleri
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_actual = np.argmax(y_test, axis=1)

# Confusion matrix ve classification report
cm = metrics.confusion_matrix(y_actual, y_pred_labels)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(metrics.classification_report(y_actual, y_pred_labels, digits=4))
