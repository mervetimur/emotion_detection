# Emotion Detection

Bu proje, görüntü veya video üzerinden yüz ifadelerini tespit edip duygu sınıflandırması yapan bir yapay zekâ uygulamasıdır.

## Proje Yapısı

- src/: Kod dosyaları
- models/: Eğitilmiş model ağırlıkları
- data/: Eğitim/test verileri
- haarcascade/: Haarcascade XML dosyaları
- README.md
- requirements.txt
- LICENSE
- .gitignore

## Kullanım

Eğitim:
python src/train_or_inference.py --data_dir data/ --save_model_path models/

Canlı Duygu Tespiti:
python src/realtime_inference.py --model_path models/model.h5


## Dataset

This project uses the FER2013 dataset, which is publicly available on Kaggle:  
https://www.kaggle.com/datasets/msambare/fer2013

Note: The dataset is not included in this repository due to its large size. Please download it manually from Kaggle.
