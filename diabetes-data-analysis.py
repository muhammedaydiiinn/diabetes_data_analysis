

import numpy as np 
import pandas as pd 
from scipy import stats 
import seaborn as sns 
import matplotlib.pyplot as plt
data= pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')
data.head()


data.info()


# Bu veriler herhangi bir boş değer içermiyor. Bu nedenle, değerleri doldurma/düşürme konusunda endişelenmemize gerek yok.

data.isnull().sum()

data.duplicated()
data=data.drop_duplicates()
data.head()


# Veriler ayrıca herhangi bir yinelenen değer içermiyor. Bu nedenle onlar için endişelenmemize gerek yok.



plt.plot(data.BMI , color ="green")
plt.xlabel("BMI")
plt.show()
plt.close()




# Hamilelik süresi ile yaş arasındaki ilişkiyi gösteren scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Pregnancies', y='Age', data=data, hue='Outcome')
plt.title('Hamilelik Süresi vs. Yaş')
plt.xlabel('Hamilelik Süresi')
plt.ylabel('Yaş')
plt.legend(title='Outcome')
plt.show()




# Hamilelik sayısı dağılımını gösteren histogram
plt.figure(figsize=(8, 6))
sns.histplot(data['Pregnancies'], bins=15, kde=True)
plt.title('Hamilelik Sayısı Dağılımı')
plt.xlabel('Hamilelik Sayısı')
plt.ylabel('Frekans')
plt.show()



data.describe()


# Tüm biyoparametreler aralık içindedir. Dolayısıyla verilerde gözlemsel/yapısal hatalar yoktur. Bu nedenle onlar için endişelenmemiz gerekiyor.


for x in data.columns:
    z=np.abs(stats.zscore(data[x]))
    print(x+str(z))


# z-skoru istatistiksel parametresine göre, yukarıdaki veri noktaları aykırı değerler olarak kabul edilir. Ancak bunların aykırı değerler olduğunu düşünmüyorum ve bu verilerin diyabet durumunu belirlemek için gerekli olduğunu düşünüyorum.

# Bir sonraki adım özellik ölçeklendirmedir. Bu veri kümesi için ML ve DL karşılaştırması yapıyorum. Makine öğrenimi için mesafeye dayalı bir algoritma olan SVM'yi düşünüyorum. Dolayısıyla verilerin normalleştirilmesi uygun olacaktır. Öte yandan, DL için, yerel minimumların daha hızlı tanımlanmasına yardımcı olabileceğinden, verilerin standartlaştırılmasının uygun olacağı gradyan iniş tabanlı bir algoritma olan YSA'yı düşünüyorum.


data_norm=data.copy()
for column in data.columns:
    data_norm[column] = (data_norm[column] - data_norm[column].min()) / (data_norm[column].max() - data_norm[column].min()) 
data_norm.head()


# Normalleştirilmiş verilerin dağılımını gösteren kutu grafiği
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_norm, orient='h')
plt.title('Normalleştirilmiş Verilerin Dağılımı')
plt.xlabel('Değer Aralığı')
plt.show()



lis=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
def standartization(x):
    x_std = x.copy(deep=True)
    for column in lis:
        x_std[column] = (x_std[column] - x_std[column].mean()) / x_std[column].std() 
    return x_std

data= standartization(data)
data.head()


# Normalleştirilmiş verilerin dağılımını gösteren histogramlar
plt.figure(figsize=(12, 8))
for column in lis:
    sns.histplot(data[column], kde=True, label=column, alpha=0.5)
    
plt.title('Normalleştirilmiş Verilerin Dağılımı')
plt.xlabel('Değer Aralığı')
plt.ylabel('Frekans')
plt.legend()
plt.show()


data.info()


data['Outcome'].value_counts()
y=data['Outcome']
x=data.drop(['Outcome'],axis=1)
yn=data_norm['Outcome']
xn=data_norm.drop(['Outcome'],axis=1)


# Standartlaştırılmış ve normalleştirilmiş veriler için ayrı train ve test setleri oluşturuyorum. n eki olanlar normalleştirilir.

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.15,stratify=y)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


from sklearn.model_selection import train_test_split
xntrain,xntest,yntrain,yntest= train_test_split(xn,yn,test_size=0.15,stratify=y)
print(xntrain.shape)
print(xntest.shape)
print(yntrain.shape)
print(yntest.shape)


plt.figure(figsize=(10, 6))

# Eğitim ve test veri dağılımını gösteren bar grafikleri
plt.bar(['Eğitim Verisi', 'Test Verisi'], [xtrain.shape[0], xtest.shape[0]], color=['blue', 'orange'])
plt.xlabel('Veri Seti')
plt.ylabel('Örnek Sayısı')
plt.title('Eğitim ve Test Veri Dağılımı')
plt.show()



from sklearn.svm import SVC
svm_model= SVC(kernel='rbf',gamma=8)
svm_model.fit(xntrain,yntrain)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
predictions= svm_model.predict(xntrain)
percentage=svm_model.score(xntrain,yntrain)
res=confusion_matrix(yntrain,predictions)
print("Training confusion matrix")
print(res)
predictions= svm_model.predict(xntest)
percentage=svm_model.score(xntest,yntest)
res=confusion_matrix(yntest,predictions)
print("validation confusion matrix")
print(res)
print(classification_report(ytest, predictions))

# check the accuracy on the training set
print('training accuracy = '+str(svm_model.score(xntrain, yntrain)*100))
print('testing accuracy = '+str(svm_model.score(xntest, yntest)*100))


# SVM'yi özellik ölçekleme olmadan ve standardizasyonla eğittim. Özellik ölçeklendirmesiz ve standardizasyonlu olarak %55 ve %62'lik test doğruluğu üretti. Dolayısıyla normalleştirme, SVM gibi mesafeye dayalı algoritmalar için iyidir.
# DL kısmı için, gizli katman olarak 256 nörondan oluşan 2 katmandan oluşan YSA'yı düşünüyorum. Daha fazla nöron ve katman göz önüne alındığında, aşırı uyum sağlandı. Dolayısıyla bu hiperparametrelerle sınırlıyım. Adam iyileştirici ve çapraz entropi kaybı işlevi kullanılarak derlendi.

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
dl_model = Sequential() 

dl_model.add(Dense(256,  activation = 'relu' ,input_shape=([8]))) #input layer
dl_model.add(Dense(256,  activation = 'relu'))
dl_model.add(Dense(1,activation = 'sigmoid'))
dl_model.summary()
dl_model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' ,metrics = ['accuracy','Precision','Recall','AUC'])



num_epochs = 50
history = dl_model.fit(xtrain ,
                    ytrain ,
                    epochs= num_epochs ,
                    steps_per_epoch=200,
                    validation_data=(xtest ,ytest))



plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()




dl_model.evaluate(xtrain,ytrain)


dl_model.evaluate(xtest,ytest)


# Gördüğünüz gibi YSA, SVM'den çok daha düşük olan %68 test doğruluğu üretti. Dolayısıyla ML algoritmasının DL algoritmasından daha iyi ürettiğini söyleyebiliriz. Bu sonuca varabilir miyiz yoksa bir şeyi mi kaçırıyoruz?

print(data['Outcome'].value_counts())
df_class_0 = data[data['Outcome'] == 0]
df_class_1 = data[data['Outcome'] == 1]


# Görüldüğü gibi sınıf dengesizliği var, diyabet negatif olanların sayısı diyabet pozitif olanların iki katı. Bu senaryoda, algoritmaların performansını doğruluğa dayalı olarak karşılaştıramayız. Bu yüzden, sınıf dengesizliğinin üstesinden gelmek için azınlık sınıfını çoğunluk sınıfının (500) örneklerine göre yüksek örnekledim. Yani toplam veri eşit dağılımlı 1000 örnekten oluşmaktadır. Bu işlemi standartlaştırılmış ve normalleştirilmiş veri kümeleri için tekrarladım.

print(data_norm['Outcome'].value_counts())
df_n_class_0 = data_norm[data_norm['Outcome'] == 0]
df_n_class_1 = data_norm[data_norm['Outcome'] == 1]


# Sınıf dağılımını gösteren bir bar plot
plt.figure(figsize=(6, 4))
sns.countplot(data['Outcome'])
plt.title('Sınıf Dağılımı')
plt.xlabel('Outcome')
plt.ylabel('Sayı')
plt.show()




df_class_1_over = df_class_1.sample(500, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
df_test_over.info()



df_n_class_1_over = df_n_class_1.sample(500, replace=True)
df_test_n_over = pd.concat([df_n_class_0, df_n_class_1_over], axis=0)
df_test_n_over.info()


y1=df_test_over['Outcome']
df_test_over=df_test_over.drop(['Outcome'],axis=1)
X1=df_test_over


y1n=df_test_n_over['Outcome']
df_test_n_over=df_test_n_over.drop(['Outcome'],axis=1)
X1n=df_test_n_over


# Parametrelerin geri kalanı aynıdır. Bu, tren testi bölme oranını ve algoritma parametrelerini içerir. Artık SVM ve YSA, üst örneklenmiş veri kümeleri kullanılarak eğitilmektedir. Daha önce olduğu gibi, SVM için normalleştirilmiş veri seti ve YSA için standartlaştırılmış veri seti.

from sklearn.model_selection import train_test_split

X1_s_train,X1_s_test ,y1_s_train, y1_s_test = train_test_split(X1,y1,
                                                   test_size=0.2,
                                                   random_state=0,
                                                  shuffle = True,
                                                  stratify = y1)

print('training data shape is :{}.'.format(X1_s_train.shape))
print('training label shape is :{}.'.format(y1_s_train.shape))
print('testing data shape is :{}.'.format(X1_s_test.shape))
print('testing label shape is :{}.'.format(y1_s_test.shape))



from sklearn.model_selection import train_test_split

X1_s_n_train,X1_s_n_test ,y1_s_n_train, y1_s_n_test = train_test_split(X1n,y1n,
                                                   test_size=0.2,
                                                   random_state=0,
                                                  shuffle = True,
                                                  stratify = y1n)

print('training data shape is :{}.'.format(X1_s_n_train.shape))
print('training label shape is :{}.'.format(y1_s_n_train.shape))
print('testing data shape is :{}.'.format(X1_s_n_test.shape))
print('testing label shape is :{}.'.format(y1_s_n_test.shape))




from sklearn.svm import SVC
svc_s_model = SVC(kernel='rbf',gamma=8)
svc_s_model.fit(X1_s_n_train, y1_s_n_train)




from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
predictions= svc_s_model.predict(X1_s_n_train)
percentage=svc_s_model.score(X1_s_n_train,y1_s_n_train)
res=confusion_matrix(y1_s_n_train,predictions)
print("Training confusion matrix")
print(res)
predictions= svc_s_model.predict(X1_s_n_test)
percentage=svc_s_model.score(X1_s_n_test,y1_s_n_test)
res=confusion_matrix(y1_s_n_test,predictions)
print("validation confusion matrix")
print(res)
print(classification_report(y1_s_n_test, predictions))
# check the accuracy on the training set
print('training accuracy = '+str(svc_s_model.score(X1_s_n_train, y1_s_n_train)*100))
print('testing accuracy = '+str(svc_s_model.score(X1_s_n_test, y1_s_n_test)*100))


# Algoritmanın doğruluğunda çok fazla bir değişiklik yok ama özellikle diyabet sınıfı için, örnekleme öncesi ve sonrası sınıflandırma raporunda büyük gelişme var.

num_epochs = 50
history = dl_model.fit(X1_s_train ,
                    y1_s_train ,
                    epochs= num_epochs ,
                    steps_per_epoch=200,
                    validation_data=(X1_s_test ,y1_s_test))




dl_model.evaluate(X1_s_train ,
                    y1_s_train)


dl_model.evaluate(X1_s_test ,y1_s_test)


# Standartlaştırılmış ve örneklenmiş veriler üzerinde eğitilen YSA, %93 test doğruluğu ile en iyi sonucu verdi. Bu defterde çok iş yaptım, umarım bu bir olumlu oyu hak eder!! Teşekkürler...
# Yanlış bir şey yaptıysam lütfen belirtin.

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim Doğruluğu')
plt.xlabel('Epok Sayısı')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()




import matplotlib.pyplot as plt

# SVM Model
svm_train_accuracy = 78  # Bu değeri gerçek SVM eğitim doğruluk oranıyla değiştirin
svm_test_accuracy = 78   # Bu değeri gerçek SVM test doğruluk oranıyla değiştirin

# Neural Network Model
nn_train_accuracy = history.history['accuracy'][-1] * 100
nn_test_accuracy = dl_model.evaluate(X1_s_test, y1_s_test)[1] * 100

# Doğruluk oranlarını listeye aktaralım
accuracy_list = [svm_train_accuracy, svm_test_accuracy, nn_train_accuracy, nn_test_accuracy]
labels = ['SVM (Eğitim)', 'SVM (Test)', 'YSA (Eğitim)', 'YSA (Test)']

# Bar grafiği oluşturalım
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracy_list, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Modeller')
plt.ylabel('Doğruluk Oranı (%)')
plt.title('Modellerin Doğruluk Oranları')
plt.ylim(0, 100)  # Y ekseni sınırlaması
plt.show()


corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='RdYlBu', annot=True, mask=mask)
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Korelasyon matrisini hesaplayın
correlation_matrix = data.corr()

# Korelasyon matrisini görselleştirin
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Veri Seti Korelasyon Heatmap')
plt.show()

