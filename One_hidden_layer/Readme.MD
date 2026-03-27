Banknot Kimlik Doğrulama - Tek Gizli Katmanlı MLP Modeli
Bu proje, banknotların gerçek mi yoksa sahte mi olduğunu sınıflandırmak için tasarlanmış, sıfırdan NumPy kütüphanesi kullanılarak yazılmış bir Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) uygulamasıdır.

Proje Hakkında
Proje kapsamında, verilen bir veri seti üzerinde tek gizli katmanlı (one hidden layer) bir yapay sinir ağı modeli eğitilmiştir. Modelin eğitim süreci, ileri yayılım (forward propagation) ve geri yayılım (backpropagation) algoritmalarını temel alarak gerçekleştirilmiştir.

Veri Seti Özellikleri
Kullanılan veri seti (BankNote_Authentication.csv), banknot görüntülerinden çıkarılan şu özellikleri içermektedir:

Varyans (Variance)
Çarpıklık (Skewness)
Basıklık (Curtosis)
Entropi (Entropy)
Sınıf (Class): 0 (Gerçek), 1 (Sahte)

Teknik Detaylar
Modelin mimarisi ve eğitim parametreleri aşağıda belirtilmiştir:

Girdi Katmanı: 4 nöron (Özellik sayısı).

Gizli Katman: 3, 5 ve 8 nöron sayıları ile farklı deneyler yapılmıştır.

Çıktı Katmanı: 1 nöron (Sigmoid aktivasyon fonksiyonu ile binary classification).

Aktivasyon Fonksiyonları: Gizli katmanda tanh, çıktı katmanında sigmoid kullanılmıştır.

Maliyet Fonksiyonu: Log-Loss (Cross-Entropy).

Optimizasyon: Gradyan İnişi (Gradient Descent).

Performans Analizi
Farklı hiperparametrelerle yapılan testler sonucunda elde edilen ortalama başarı metrikleri:
Accuracy (Doğruluk): %98.5+
F1 Score: 0.98

Eğitim Süreci: Maliyet grafiği, modelin iterasyonlar arttıkça hata payını minimize ettiğini ve optimum noktaya ulaştığını göstermektedir.

Önemli Not: Bu projede hazır derin öğrenme kütüphaneleri (PyTorch, TensorFlow) veya hazır sınıflandırıcı modelleri 
(sklearn.neural_network.MLPClassifier) kullanılmamıştır. Tüm ileri yayılım ve geri yayılım algoritmaları, sinir ağı mantığını tam olarak yansıtmak adına 
NumPy kütüphanesi ile temel seviyede (from scratch) kodlanmıştır.