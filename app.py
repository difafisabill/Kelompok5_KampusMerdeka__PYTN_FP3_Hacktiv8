import os
import pickle
import shutil


import pandas as pd
import requests
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# dataset dan model
csv_url = 'https://github.com/difafisabill/Kelompok5_KampusMerdeka__PYTN_FP3_Hacktiv8/raw/main/data_heart_Cleaned.csv'
model_url = 'https://github.com/difafisabill/Kelompok5_KampusMerdeka__PYTN_FP3_Hacktiv8/raw/main/models_ensemble_bagging.pkl'

def download_model_from_url(model_url, save_path):
    if model_url.startswith('http'):
        response = requests.get(model_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        shutil.copy(model_url, save_path)

model_path = 'model.pkl'
download_model_from_url(model_url, model_path)

if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error("Failed to load Logistic Regression model.")





# Tampilan
st.markdown("# Prediksi Gagal Jantung")
st.markdown("Penyakit Cardiovascular (CVD) merupakan penyebab kematian nomor 1 di dunia. Sekitar 17,9 juta jiwa atau 31% kematian disebabkan oleh CVD. Gagal jantung merupakan kejadian yang umum disebabkan oleh penyakit kardiovaskular. Mayoritas penyakit kardiovaskular dapat dicegah dengan mengatasi faktor-faktor penyebab, seperti rokok, pola makan yang tidak sehat, aktivitas fisik yang kurang, dan konsumsi alkohol. Sehingga prediksi gagal jantung perlu dilakukan untuk menghindari risiko yang lebih parah")


def main():
    st.sidebar.title('Prediksi Gagal Jantung')

    @st.cache_resource
    def load_data():
        data = pd.read_csv(csv_url)
        return data
      
    def indexInput(input):
        input_mapping={
            'Ya':1,
            'Tidak':0
        }
        return input_mapping.get(input, 0)
    
    def indexSex(sex):
        sex_mapping={
            'Laki-laki':1,
            'Perempuan':0
        }
        return sex_mapping.get(sex, 0)

    def heart_failure(predict):
        if predict == 0:
            return f'Sehat'
        else:
            return f'Indikasi gagal jantung'

    def calculate_recovery_potential(creatinine_phosphokinase, platelets,serum_creatinine, serum_sodium, age, ejection_fraction):
        # Hitung kolom yang dapat mengukur tingkat keparahan penyakit
        severity = creatinine_phosphokinase / platelets
        severity += serum_creatinine / serum_sodium
        severity += age / ejection_fraction

        # Hitung kolom yang dapat mengukur potensi pemulihan
        recovery_potential = 1 / severity

        return recovery_potential

    # Fungsi untuk scaling data
    def scale_data(data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    data = load_data()
    check_box = st.sidebar.checkbox("Show Dataset")
    if (check_box):
        st.markdown("#### Heart Diseases Dataset")
        st.write(data)

    age = st.sidebar.number_input('Age',
                                min_value= 35,
                                step=1)
    anemia = st.sidebar.radio('Menderita Anemia', ('Ya', 'Tidak'), horizontal=True)
    index_anemia = indexInput(anemia)
    
    creatinine_phosphokinase = st.sidebar.slider('Kadar enzim CPK dalam darah (mcg/L)', 1, 8000, 23)
    diabetes = st.sidebar.radio('Menderita Diabetes', ('Ya', 'Tidak'), horizontal=True)
    index_diabetes = indexInput(diabetes)

    ejection_fraction = st.sidebar.slider('Persentase darah yang meninggalkan jantung pada setiap kontraksi', 1, 100, 10)
    high_blood_pressure = st.sidebar.radio('Menderita Hipertensi', ('Ya', 'Tidak'), horizontal=True)
    index_hipertensi = indexInput(high_blood_pressure)

    platelets = st.sidebar.slider('Trombosit dalam darah (kiloplatelet/mL)', 1.0, 850000.0, 25100.0)
    serum_creatinine = st.sidebar.slider('Kadar kreatinin serum dalam darah (mg/dL)', 0.0, 10.0, 0.1)
    serum_sodium = st.sidebar.slider('Kadar sodium serum dalam darah (mEq/L)', 0.0, 150.0, 100.0)
    sex = st.sidebar.radio('Jenis Kelamin', ('Laki-laki', 'Perempuan'))
    index_sex = indexSex(sex)

    smoking = st.sidebar.radio('Perokok', ('Ya', 'Tidak'), horizontal=True)
    index_smoking = indexInput(smoking)

    time = st.sidebar.slider('Periode tindak lanjut', 1, 300, 1)
    recovery_potential = calculate_recovery_potential(creatinine_phosphokinase, platelets,serum_creatinine, serum_sodium, age, ejection_fraction)

    def report_display():
        important_feature={
            'age' : age,
            'anemia' : index_anemia,
            'creatinine_phosphokinase' : creatinine_phosphokinase,
            'diabetes' : index_diabetes,
            'ejection_fraction' : ejection_fraction,
            'high_blood_pressure' : index_hipertensi,
            'platelets' : platelets,
            'serum_creatinine' : serum_creatinine,
            'serum_sodium' : serum_sodium,
            'sex' : index_sex,
            'smoking' : index_smoking,
            'time' : time,
            'recovery_potential' : recovery_potential
        }
        report_data = pd.DataFrame(important_feature, index=[0])
        return report_data

    user_feature = report_display()

    st.markdown('#### Report')
    st.write(user_feature)

    data = pd.DataFrame(user_feature)
    inputs= scale_data(data)

    if st.button('Classify'):
        st.success(heart_failure(model.predict(inputs)))


if __name__ == '__main__':
    main()
