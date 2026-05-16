# Garbage Classification

Günümüzde hızla artan kentleşme ve tüketim alışkanlıkları, atık yönetimini küresel bir sorun haline
getirmiştir. Atıkların kaynağında doğru sınıflandırılması, geri dönüşüm süreçlerinin verimliliği için kritik
öneme sahiptir. Bu çalışmada, çevresel atıkları türlerine göre sınıflandırmak amacıyla derin öğrenme
tabanlı bir sistem geliştirilmiştir. Çalışmada Trashbox ve Garbage Dataset veri setleri birleştirilerek
dengeli bir veri kümesi oluşturulmuştur. Elde edilen veri seti üzerinde DenseNet-201 ve EfficientNetV2
mimarilerinin performansları karşılaştırılmıştır. Ayrıca eğitim sürecinde farklı optimizasyon algoritmaları
ve kayıp fonksiyonlarının etkileri incelenmiştir.

---


## Proje Ekibi
* **Zeynep Palabıyık** - 220202016
* **Sena Köseoğlu** - 220202042

## Veri Seti ve Ön İşleme
Projede literatürdeki **"Trashbox"** ve **"Garbage Dataset"** veri setleri birleştirilerek **merged_dataset2** oluşturulmuştur.

* **Toplam Görüntü:** 25.100+ Adet
* **Sınıflar (6 Adet):** Cardboard, Clothes, Glass, Metal, Paper, Plastic
* **Dağılım:** %80 Eğitim, %10 Doğrulama, %10 Test
* **Ön İşleme (Preprocessing):**
    * **Resize:** Tüm görüntüler `224x224` piksel boyutuna indirgenmiştir.
    * **Normalization:** ImageNet standartları (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) kullanılarak normalize edilmiştir.

---
## Projenin Amacı ve Kapsamı

Projenin temel amacı, geri dönüşüm süreçlerini otomatikleştirebilecek yüksek
doğruluklu ve genellenebilir bir görüntü sınıflandırma modeli geliştirmektir.
Bu doğrultuda:

- Transfer Learning (Transfer Öğrenme) yöntemleri kullanılmıştır.
- Sınıf dengesizliğinin etkilerini azaltmak amacıyla **Cross Entropy Loss**
  ve **Focal Loss** fonksiyonları denenmiş, performansları analiz edilmiştir.
- Model eğitimi sürecinde **Adam** ve **AdamW** optimizasyon algoritmaları
  kullanılarak karşılaştırmalı değerlendirmeler yapılmıştır.
- Google Colab ortamında proje gerçekleştirilmiştir 

---

## Kodlar

Proje dizininde yer alan dosyalar ve işlevleri aşağıda açıklanmıştır.

### `densenet201.ipynb`

- Projenin başlangıç aşamasında yapılan çalışmaları içermektedir.
-  **ResNet-50** mimarisi ile başlamakta; devamında **DenseNet201** mimarisine geçiş yapılmaktadır
- DenseNet201 mimarisi kullanılarak yüksek model kapasitesinin
  performansa ve aşırı öğrenme (overfitting) üzerindeki etkileri analiz edilmiştir.
- Farklı optimizasyon algoritmaları (Adam, AdamW, SGD, RMSprop, NAdam)
  karşılaştırılmıştır.
- Cross Entropy, Label Smoothing ve Focal Loss gibi farklı kayıp
  fonksiyonlarının model başarısına etkileri değerlendirilmiştir.

### `densenet121.ipynb`

- Daha dengeli ve genellenebilir bir model elde etmek amacıyla
  DenseNet121 mimarisi kullanılmıştır.
- Sınıf dengesizliği problemini yönetmek amacıyla Focal Loss fonksiyonu
  tercih edilmiştir.
- Eğitim süresi ve doğruluk açısından DenseNet201 modeli ile
  karşılaştırmalı sonuçlar elde edilmiştir.

### `EfficientNetV2-2.deneme.ipynb`

- EfficientNetV2-S mimarisi kullanılarak daha hafif, hızlı ve kararlı
  bir model elde edilmesi hedeflenmiştir.
- Feature extractor katmanlarının dondurulması, Dropout ve
  Weight Decay gibi düzenlileştirme teknikleri uygulanmıştır.


### `test_image.py`

- Eğitilmiş en iyi model dosyaları (`.pth`) kullanılarak
  tek bir görüntü üzerinde sınıflandırma yapılmasında kullanılır.
- DenseNet121, DenseNet201 ve EfficientNetV2-S modelleri test edilir.
- Yorum satırları kullanılarak test edilecek model kolayca seçilebilmektedir.

---

## Kullanılan Teknikler ve Yöntemler

### Model Mimarileri

- EfficientNetV2-S
- DenseNet121
- DenseNet201

Tüm modellerde ImageNet üzerinde önceden eğitilmiş ağırlıklar kullanılmıştır.

### Optimizasyon ve Düzenlileştirme

- **Kayıp Fonksiyonları:** Cross Entropy, Label Smoothing, Focal Loss (Gamma = 2.0)
- **Optimizasyon Algoritmaları:** Adam, AdamW (Weight Decay ile), SGD, RMSprop, NAdam
- **Early Stopping:** Doğrulama kaybının iyileşmemesi durumunda eğitimin durdurulması
- **Dropout ve Batch Normalization:** Aşırı öğrenmenin azaltılması ve eğitim kararlılığının artırılması

---

## Gereksinimler

Projenin çalıştırılabilmesi için aşağıdaki Python kütüphaneleri gereklidir:

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Tqdm

---
## Kurulum ve Kullanım 

Bu proje Google Colab üzerinde geliştirilmiştir. Çalıştırmak için adımlar:

1. Dosyaları İndirin ve Notebook'u Açın: İncelemek istediğiniz dosyayı (densenet121.ipynb veya EfficientNetV2-2.deneme.ipynb) Google Colab'e yükleyin.

2. Veri Setini Bağlayın: Proje, veri setini Google Drive üzerinden çeker; Drive bağlantısını sağlayın ve dosya yollarını kontrol edin.

3. Eğitimi Başlatın: Hücreleri sırasıyla (Run All) çalıştırarak model eğitimini, kayıp grafiklerini ve test sonuçlarını gözlemleyebilirsiniz.

4. test_image.py scripti geliştirilmiştir. Bu script, DenseNet-121, DenseNet-201 ve EfficientNetV2-S mimarileri ile eğitilmiş ve 220202016_220202042_kodlar/test_bestptler/ dizini içerisine önceden kaydedilmiş model ağırlıklarını (.pth) yükleyerek, verilen bir görüntünün hangi atık sınıfına ait olduğunu tahmin eder ve bu tahmine ait güven skorunu hesaplar. Test işlemi, kodlar/test_bestptler/ dosyasında tanımlı olan hazır örnek görseller kullanılarak doğrudan gerçekleştirilebilir. Script, VS Code veya terminal üzerinden python test_image.py komutu ile çalıştırılabilir ve sonuçlar; kullanılan model türü, model ağırlık dosyası, tahmin edilen sınıf ve yüzde cinsinden güven skoru şeklinde ekrana yazdırılır.

## Sonuç ve Değerlendirme

Elde edilen modeller, test veri seti üzerinde sınıflandırma raporu
(Precision, Recall, F1-Score) ve Confusion Matrix kullanılarak değerlendirilmiştir.
Ayrıca eğitim ve doğrulama süreçlerinde modelin en iyi çıktısı 
(Loss) grafikleri sunulmuştur.

Yapılan karşılaştırmalı analizler sonucunda, aşırı öğrenme (overfitting)
problemini en iyi yöneten ve en kararlı performansı gösteren model belirlenmiştir.

