# CIFAR-10 CNN Sınıflandırma Projesi

## 1. Giriş (Introduction)

Derin öğrenme, özellikle bilgisayarla görme (computer vision) alanında son yıllarda önemli gelişmeler sağlamıştır. Bu gelişmelerin temelinde evrişimli sinir ağları (Convolutional Neural Networks - CNN) bulunmaktadır. CNN mimarileri, görüntülerden otomatik olarak anlamlı özellikler çıkarabilmesi sayesinde sınıflandırma problemlerinde yüksek başarı elde etmektedir.

Bu projede, CIFAR-10 veri seti kullanılarak farklı CNN tabanlı modellerin performansları karşılaştırılmıştır. Amaç, farklı mimarilerin aynı veri seti üzerindeki başarılarını analiz etmek ve hangi yaklaşımın daha etkili olduğunu ortaya koymaktır.

CIFAR-10 veri seti 10 sınıftan oluşmaktadır:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## 2. Yöntem (Method)

### Veri Seti

Bu çalışmada CIFAR-10 veri seti kullanılmıştır:
- Eğitim verisi: 50.000 görüntü
- Test verisi: 10.000 görüntü
- Görüntü boyutu: 32x32
- Kanal sayısı: 3 (RGB)
- Sınıf sayısı: 10

### Ön İşleme (Preprocessing)

Veri setine aşağıdaki ön işlemler uygulanmıştır:
- Normalize (mean ve std değerleri ile)
- Random Horizontal Flip
- Random Crop
- Tensor dönüşümü

Bu işlemler modelin genelleme yeteneğini artırmak için uygulanmıştır.

---

### Model 1: LeNet Benzeri CNN

Bu model klasik LeNet-5 mimarisine benzer şekilde tasarlanmıştır.

Katmanlar:
- Conv2D
- ReLU
- MaxPooling
- Fully Connected Layers

Bu model temel bir CNN mimarisi olarak referans alınmıştır.

---

### Model 2: BatchNorm + Dropout CNN

Model 1'in geliştirilmiş versiyonudur.

Eklenen katmanlar:
- Batch Normalization → eğitim stabilitesi
- Dropout → overfitting azaltma

---

### Model 3: ResNet18

ResNet18 literatürde yaygın kullanılan derin bir CNN mimarisidir.

Avantajları:
- Residual bağlantılar
- Gradient kaybını önleme
- Daha derin öğrenme kapasitesi

---

### Model 4: Hibrit Model (CNN + Random Forest)

Bu modelde:
- ResNet18 feature extractor olarak kullanılmıştır
- CNN'in son katmanı çıkarılmıştır
- Elde edilen feature'lar `.npy` olarak kaydedilmiştir
- Random Forest algoritması ile sınıflandırma yapılmıştır

---

### Model 5: Tam CNN Karşılaştırması

ResNet18 modeli hem hibrit yaklaşım hem de tam CNN olarak karşılaştırılmıştır.

---

## 3. Deneysel Kurulum (Experimental Setup)

Kullanılan hiperparametreler:

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epoch: 5

---

## 4. Sonuçlar (Results)

### Model Performansları

| Model | Test Accuracy |
|------|-------------|
| Model 1 - LeNet-like CNN | 0.5951 |
| Model 2 - Improved CNN | 0.5722 |
| Model 3 - ResNet18 | 0.7253 |
| Model 4 - Hybrid (CNN + RF) | 0.3072 |

### Grafikler

Aşağıdaki grafikler `outputs/` klasöründe bulunmaktadır:

- Loss grafikleri
- Accuracy grafikleri
- Confusion matrix görselleri

Örnek:

![Model1 Accuracy](outputs/Model1_LeNetLikeCNN_accuracy.png)
![Model3 Accuracy](outputs/Model3_ResNet18_accuracy.png)

---

### Confusion Matrix

Her model için confusion matrix analizleri yapılmıştır. Bu matrisler modelin hangi sınıflarda daha başarılı veya başarısız olduğunu göstermektedir.

---

## 5. Tartışma (Discussion)

### En iyi model: ResNet18 (%72.53)

ResNet18 modeli en yüksek doğruluğu sağlamıştır. Bunun nedeni:
- Derin mimari yapısı
- Residual bağlantılar sayesinde gradient kaybının önlenmesi

---

### Model 1 vs Model 2

Beklenen:
- Model 2 daha iyi olmalıydı

Gerçek:
- Model 1 (%59.5) > Model 2 (%57.2)

Sebep:
- Düşük epoch sayısı
- Dropout erken öğrenmeyi zorlaştırdı

---

### Hibrit Model Analizi

Hibrit model düşük performans göstermiştir (%30.72)

Sebep:
- CNN end-to-end öğrenmedi
- Feature extraction yeterli olsa da sınıflandırma zayıf kaldı

---

### Genel Değerlendirme

- Basit CNN → orta performans
- Geliştirilmiş CNN → beklenenden düşük performans
- Derin CNN → en iyi performans
- Hibrit model → alternatif ama zayıf

---

## 6. Referanslar (References)

- Krizhevsky, A. CIFAR-10 Dataset
- LeCun, Y. LeNet-5
- He, K. ResNet
- PyTorch Documentation
- Torchvision Models

---

## 7. Çalıştırma

```bash
pip install -r requirements.txt
python src/main.py --epochs 5
```

---

## 8. Proje Yapısı

```
CNN_project/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── data/
│   └── outputs/
│       ├── *.png
│       ├── *.txt
│       └── *.npy
```
