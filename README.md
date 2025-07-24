# 🚢 Titanic EDA ve ML Projesi

Bu proje, **Titanic** veri seti üzerinde yapılan **Keşifsel Veri Analizi (EDA)** ve **temel makine öğrenmesi modellemeleri** içermektedir. Projenin amacı, yolcuların hayatta kalma durumunu çeşitli değişkenlere göre analiz etmek ve sınıflandırma modelleri ile bu durumu tahmin etmektir.

## 🔍 Kullanılan Adımlar

### 1. 📊 Keşifsel Veri Analizi (EDA)
- Kategorik ve sayısal değişkenlerin görselleştirilmesi (`bar_plot`, `hist_plot`)
- Pclass, Sex, SibSp, Parch gibi değişkenlerle hayatta kalma oranları incelendi
- Korelasyon matrisi ve heatmap’lerle değişken ilişkileri yorumlandı
- Outlier (aykırı değer) tespiti ve temizliği (IQR yöntemi)

### 2. 🛠️ Eksik Verilerin Doldurulması
- **Age** sütunu: Lineer regresyon modeliyle tahmin edilerek dolduruldu
- **Fare** sütunu: Ortalama ile dolduruldu
- **Embarked** sütunu: Kutucuk grafiği analizine göre "C" ile dolduruldu

### 3. 🧪 Feature Engineering (Yeni Özellikler)
- **Title**: İsimlerden unvan çıkarımı (Mr, Mrs, Miss vs.)
- **Family Size**: `Parch + SibSp + 1` ile oluşturuldu, küçük/kalabalık aile ayrımı yapıldı
- **Embarked, Sex, Pclass, Ticket** gibi sütunlar one-hot encoding ile dönüştürüldü
- **Gereksiz sütunlar** (Name, PassengerId, Ticket, Cabin) çıkarıldı

### 4. 🧠 Modelleme ve Değerlendirme
Kullanılan sınıflandırma modelleri:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVC)
- K-Nearest Neighbors (KNN)

#### Modelleme detayları:
- Veriler `train_test_split` ile ayrıldı
- `StandardScaler` ile bazı modellerde veriler ölçeklendirildi
- `GridSearchCV` ve `StratifiedKFold` ile hiperparametre optimizasyonu yapıldı
- Tüm modellerin **cross-validation** skorları kıyaslandı

#### Ensemble Model:
- **VotingClassifier (soft voting)** kullanıldı (Random Forest + Decision Tree + Logistic Regression)
- Final test tahminleri bu model üzerinden yapıldı

## 📁 Klasör Yapısı

