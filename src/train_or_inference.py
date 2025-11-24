import cv2
import numpy as np
import tensorflow as tf

# 1. Modeli Yükle
model_path = 'C:/Users/Hp/Desktop/MY_PR/python.py/son/emotion_detection_model.h5'
model = tf.keras.models.load_model(model_path)

# 2. Duygu Etiketleri
emotion_labels = {0: 'Mutlu', 1: 'Notr', 2: 'Uzgun', 3:'Kizgin', 4:'Saskin', 5:'Igrenme', 6: 'Korku'}

# 3. Kamera Açma
cap = cv2.VideoCapture(0)

# Uzun süreli açık kalması için 30 saniye bekle
wait_time = 30  # bekleme süresi (saniye)
start_time = cv2.getTickCount()  # Başlangıç zamanı

while True:
    # Kamera görüntüsünü al
    ret, frame = cap.read()
    if not ret:
        break

    # AYNALAMA buraya eklendi
    frame = cv2.flip(frame, 1)

    # Yüz tespiti için Haarcascade sınıflandırıcısı yükle
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Görüntüyü gri tonlamaya çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Yüz bölgesini kes
        face = gray_frame[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Duygu tespiti yap
        predictions = model.predict(face)
        emotion_index = np.argmax(predictions[0])

        emotion_label = emotion_labels.get(emotion_index, "Bilinmeyen")

        # Dikdörtgen çiz ve duygu durumunu yazdır
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Sonucu göster
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time > wait_time:
        break

cap.release()
cv2.destroyAllWindows()
