import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt
import joblib

# Veriyi oku
data = pd.read_csv('diabetes.csv')

# Değişkenleri ayır
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = data[feature_cols] 
y = data['Outcome']

# Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # %70 eğitim %30 test


# Entropiye göre dallanan, maksimum derinliği 4 olan bir karar ağacı modeli oluştur ve eğit
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(X_train,y_train)


# Test veri kümesi için modelden tahmin al ve gerçek sonuçlarla model tahminlerini karşılaştır
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Modeli kaydet
joblib.dump(clf,'decisiontree.pkl')  

# Eğitilen karar ağacı modelinin görsel olarak çiz
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=feature_cols, class_names=["Negative", "Positive"], filled=True,fontsize=14)
plt.show()
