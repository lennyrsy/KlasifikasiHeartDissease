import streamlit as st
import pandas as pd
import pickle

dataset = pd.read_csv('HeartDisease.csv')

st.title('Dashboard Klasifikasi Penyakit Jantung')

try:
    umurs = (st.text_input('Masukkan Umur'))
    umur = int(umurs)

    # sex = st.text_input('Masukkan Jenis Kelamin')
    sexs = st.radio(
        'Jenis Kelamin',
        ('Laki-laki', 'Perempuan',)
    )
    if sexs == 'Laki-laki':
        sex = 1
    else:
        sex = 0

    # chest = st.text_input('Masukkan Chest Pain Type')

    chests = st.radio(
        'Chest Pain Type',
        ('tipikal', 'nontipikal', 'bukan nyeri dada', 'asimtotik')
    )
    if chests == 'tipikal':
        chest = 1
    elif chests == 'nontipikal':
        chest = 2
    elif chests == 'bukan nyeri dada':
        chest = 3
    else:
        chest = 4

    fastings = int(st.text_input('Masukkan Fasting Blood Sugar'))
    fasting = 0 if fastings <= 120 else 1

    # excercise = st.text_input('Masukkan Excercise Angina')
    excercise = 0
    excercises = st.radio(
        'Excercise Angina',
        ('Ya', 'Tidak')
    )
    if excercises == 'Ya':
        excercise = 1
    else:
        excercise = 0

    old_peak = float(st.text_input('Masukkan Old Peak'))
    # st_slope = st.text_input('Masukkan ST Slope')

    st_slopes = st.radio(
        'ST Slope',
        ('Naik', 'Mendatar', 'Menurun')
    )
    if st_slopes == 'Naik':
        st_slope = 1
    elif st_slopes == 'Mendatar':
        st_slope = 2
    else:
        st_slope = 3

    new_val = pd.DataFrame([[umur, sex, chest, fasting, excercise, old_peak, st_slope]])

    infile = open('model_svm.pkl', 'rb')
    svm_model = pickle.load(infile)
    infile.close()

    infile = open('enc.pkl', 'rb')
    encoding = pickle.load(infile)
    infile.close()

    new_val = encoding.transform(new_val)


    y_pred = svm_model.predict(new_val)


    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        text-align: center;
        color: #270;
        background-color: #DFF2BF;
    }
    </style>
    """, unsafe_allow_html=True)

    if y_pred == 0:
        hasil = ('Tidak Terkena Penyaki Jantung')
    else:
        hasil = ('Terkena Penyakit Jantung')

    st.markdown(f'<p class="big-font">Hasil Klasifikasi : {hasil}</p>', unsafe_allow_html=True)
except:
    pass
