import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Modeli yükle
clf = joblib.load('decisiontree.pkl')

# Kullanıcıdan veri girişi almak
st.title("Karar Ağacı ile Diyabet Tahmini")

col1, col2 = st.columns(2)
# Kullanıcıya inputlar sağla
with col1:
 pregnancies = st.slider("Gebelik sayısı", 0, 15)
 glucose = st.slider("Glikoz", 0, 200, 100)
 blood_pressure = st.slider("Kan basıncı", 0, 150, 80)
 skin_thickness = st.slider("Cilt kalınlığı", 0, 50, 20)
with col2:
 insulin = st.slider("İnsülin", 0, 800, 100)
 bmi = st.slider("BMI (Vücut kitle endeksi)", 0.0, 50.0, 25.0)
 pedigree_function = st.slider("Diyabet Pedigri Fonksiyonu", 0.0, 2.5, 0.5)
 age = st.slider("Yaş", 18, 100)

# Kullanıcıdan alınan verileri bir DataFrame'e dönüştür
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]],
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

if st.button('Tahmin et'):
# Karar ağacını kullanarak tahmin yap
    prediction = clf.predict(input_data)

# Sonucu kullanıcıya göster
    if prediction == 1:
        st.write("Tahmin: Pozitif (Diyabet tespit edildi.)")
    else:
        st.write("Tahmin: Negatif (Diyabet Yok.)")

# Karar ağacını görselleştir
st.subheader("Decision Tree Visualization")
fig = plt.figure(figsize=(24,16))
plot_tree(clf, feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
          class_names=["Negative", "Positive"], filled=True,fontsize=13)
st.pyplot(fig)
