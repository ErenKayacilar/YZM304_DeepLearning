1. Temel Tanımlar

Matris Manipülasyonu:
Matrislerin oluşturulması, dönüştürülmesi ve çarpanlarına ayrıştırılması gibi işlemlerin bütünüdür. Makine öğrenmesinde veriler n×m boyutlu matrisler olarak temsil edilir; bu matrisler üzerinde yapılan cebirsel işlemler modelin eğitilmesini ve yorumlanmasını mümkün kılar.

Özdeğer (Eigenvalue): 
Kare bir A matrisi için, sıfırdan farklı bir v vektörü ve λ skaleri A·v = λ·v eşitliğini sağlıyorsa, λ bu matrisin bir özdeğeridir. Bir özvektör, A matrisiyle çarpıldığında yalnızca ölçeği değişen (λ katı kadar), yönü değişmeyen özel bir vektördür; λ ise bu ölçekleme miktarını veren özdeğerdir.

 Özvektör (Eigenvector): Özvektörler yeni özellik uzayının yönlerini, özdeğerler ise bu yönlerdeki büyüklüğü (varyans miktarını) belirler. 

2. Makine Öğrenmesi ile İlişkisi:
 Makine öğrenmesinde verinin içindeki gizli yapıyı açığa çıkarmak için matris ayrıştırma yöntemleri kullanılır. Bu noktada özdeğer ve özvektörler merkezi bir rol üstlenir.

 Kovaryans Matrisi: 
 Kovaryans matrisinin köşegen elemanları değişkenlerin varyansını, köşegen dışı elemanlar ise değişkenler arasındaki artıklığı (kovaryansı) gösterir. Boyut azaltmada amaç varyansı maksimize ederken kovaryansı minimize etmektir; bu da kovaryans matrisinin köşegenleştirilmesiyle, yani özdeğer ayrıştırmasıyla elde edilir.

PCA (Temel Bileşen Analizi):
PCA, verinin kovaryans matrisinin özvektörlerini ve özdeğerlerini bulur. En büyük özdeğere karşılık gelen özvektör, verideki en yüksek varyans yönünü gösterir ve birinci temel bileşeni tanımlar; ardışık bileşenler birbirlerine dik biçimde azalan varyansları yakalar. 


3. Kullanılan Yöntem ve Yaklaşımlar

Makine öğrenmesinde verinin içindeki gizli yapıyı açığa çıkarmak için matris ayrıştırma yöntemleri kullanılır. Bu noktada özdeğer ve özvektörler merkezi bir rol üstlenir.

PCA-EIG (Özdeğer Ayrıştırması ile PCA)
Standardize edilmiş veri setinin kovaryans matrisi hesaplanır; bu matris özvektörler ve özdeğerlere ayrıştırılarak verinin yayıldığı yönler ve büyüklükler elde edilir.Ancak bu yöntem simetrik ve kare matrislerde çalışır ve sayısal açıdan kararsız olabilir.

PCA-SVD (Tekil Değer Ayrıştırması ile PCA)
Herhangi bir m×n matrisi X = U · Σ · Vᵀ biçiminde üç matrisin çarpımına ayrıştırılır. SVD, özdeğer ayrıştırmasının aksine kare olmayan matrislere de uygulanabilir; ayrıca tekil değerler sayısal olarak özdeğerlerden daha kararlıdır.Bu nedenle PCA-SVD, günümüz makine öğrenmesi kütüphanelerinde çok daha yaygındır

LDA (Doğrusal Ayırt Edici Analiz)
PCA tüm veri setini farklı bir özellik alt uzayına yansıtırken, LDA sınıflar arasındaki ayrımı maksimize eden yönleri bulmaya çalışır; bu da örüntü sınıflandırma problemlerinde kullanışlıdır

Öneri Sistemleri ve Görüntü İşleme
SVD; veri sıkıştırma, gürültü giderme, yüz tanıma ve öneri sistemlerinde işbirlikçi filtreleme gibi pek çok uygulamada kullanılmaktadır.

4. Kaynaklar

Hui, J. (2020). Machine Learning — Singular Value Decomposition (SVD) & Principal Component Analysis (PCA). Medium. https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491

ML and DL Explained. (2025). Singular Value Decomposition (SVD) vs. Eigen Decomposition: A Deep Dive. Medium. https://medium.com/@ml_dl_explained/singular-value-decomposition-svd-vs-eigen-decomposition-a-deep-dive-6cc99463b45d


5. numpy.linalg.eig Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi:

numpy.linalg.eig(a)
```

| Parametre | Tür | Açıklama |
|-----------|-----|----------|
| `a` | `(..., M, M) array` | Özdeğerleri ve özvektörleri hesaplanacak **kare matris** |

**Dönüş değerleri** bir `namedtuple` olarak gelir:

| Çıktı | Tür | Açıklama |
|-------|-----|----------|
| `eigenvalues` | `(..., M) array` | Her biri katlılığı kadar tekrarlanan özdeğerler |
| `eigenvectors` | `(..., M, M) array` | Normalize edilmiş (birim uzunluklu) sağ özvektörler; `eigenvectors[:,i]` → `eigenvalues[i]`'e karşılık gelir |

> **Not:** Özdeğerler sıralı gelmez. Sonuç, sanal kısım sıfır değilse karmaşık (`complex`) türde olur; sıfırsa otomatik olarak gerçek (`float`) türe dönüştürülür.

---

### 2. Kaynak Kod Katmanları

`numpy.linalg.eig` üç katmandan oluşur:
```
Python Arayüzü (_linalg.py)
        ↓
C Uzantısı (umath_linalg.c)
        ↓
LAPACK Rutinleri (Fortran)
    ├── dgeev  → gerçel matrisler
    └── zgeev  → karmaşık matrisler
```

#### 2.1 Python Katmanı (`numpy/linalg/_linalg.py`)

Kaynak kodu `v2.1.0` etiketindeki GitHub deposunda `numpy/linalg/_linalg.py` dosyasında, `#L1345–L1499` satır aralığında yer alır. Bu katman şunları yapar:

1. Girdi matrisinin kare olup olmadığını kontrol eder (`_assert_stacked_square`)
2. Veri tipini belirler: gerçel mi, karmaşık mı? (`_commonType`)
3. Uygun LAPACK rutin sarmalayıcısını (`_umath_linalg.eig`) çağırır
4. Özdeğerleri ve özvektörleri temizleyip döndürür

#### 2.2 LAPACK Katmanı (Asıl Hesaplama)

`eig` fonksiyonu, gerçel kare matrisler için `dgeev`, karmaşık kare matrisler için `zgeev` LAPACK rutinlerine basit bir arayüzdür. 

LAPACK rutinleri şu adımları izler:

**Adım 1 — Hessenberg Dönüşümü:**
Matris, Householder yansımaları ile üst Hessenberg formuna (`H`) indirgenir. Bu işlem özdeğerleri korur ancak sonraki adımı çok daha verimli hale getirir.
```
A  →  Q · H · Qᵀ

Adım 2 — QR İterasyonu:
Hessenberg matrisine QR algoritması uygulanarak Schur formuna ulaşılır. QR algoritması, matrisi köşegende özdeğerlerin yer aldığı üst üçgen bir forma yakınsar.

Adım 3 — Özvektörlerin Geri Dönüşümü:
Schur vektörleri, Hessenberg dönüşümünün tersi alınarak orijinal matrisin özvektörlerine dönüştürülür ve ardından normalize edilir.

3. Matematiksel Temel
eig fonksiyonu sağ özvektörleri hesaplar: a @ eigenvectors[:,i] = eigenvalues[i] * eigenvectors[:,i] denklemi tüm i değerleri için sağlanır.
Tüm özdeğerler birbirinden farklıysa özvektörler teorik olarak doğrusal bağımsızdır ve özvektörler matrisi kullanılarak inv(eigenvectors) @ a @ eigenvectors benzerlik dönüşümü ile A köşegenleştirilebilir
Sol özvektörler bu fonksiyon tarafından hesaplanmaz. Bir matrisin sol ve sağ özvektörleri genel olarak birbirinin (eşlenik) transpozu değildir.Sol özvektörler gerekiyorsa scipy.linalg.eig(a, left=True) kullanılmalıdır

4. Kod Örneği (Dokümantasyondan)

import numpy as np

A = [[6,  1, -1],
     [0,  7,  0],
     [3, -1,  2]]

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Özdeğerler:", eigenvalues)
# → [3. 5. 7.]

print("Özvektörler (sütunlar):")
print(eigenvectors)

# Doğrulama: A @ v = λ · v
for i in range(len(eigenvalues)):
    lhs = np.array(A) @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    print(f"λ{i+1} doğrulama farkı: {np.allclose(lhs, rhs)}")






