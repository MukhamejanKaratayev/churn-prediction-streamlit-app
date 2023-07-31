import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle


st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# global variables
df_predicted = pd.DataFrame()
uploaded_file = None

# session stare initialization
if 'df_input' not in st.session_state or uploaded_file is None:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

# ML section start
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender',
'seniorcitizen',
'partner',
'dependents',
'phoneservice',
'multiplelines',
'internetservice',
'onlinesecurity',
'onlinebackup',
'deviceprotection',
'techsupport',
'streamingtv',
'streamingmovies',
'contract',
'paperlessbilling',
'paymentmethod']
le_enc_cols = ['gender', 'partner', 'dependents','paperlessbilling', 'phoneservice']
gender_map = {'male': 0, 'female': 1}
y_n_map = {'yes': 1, 'no': 0}

# logistic regression model
model_file_path = 'models/lr_model_churn_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))

# encoding model DictVectorizer
encoding_model_file_path = 'models/encoding_model.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

@st.cache_data
def predict_churn(df_input, treshold):

    scaler = MinMaxScaler()

    df_original = df_input.copy()
    df_input[numerical] = scaler.fit_transform(df_input[numerical])

    for col in le_enc_cols:
        if col == 'gender':
            df_input[col] = df_input[col].map(gender_map)
        else:
            df_input[col] = df_input[col].map(y_n_map)

    dicts_df = df_input[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    # X[np.isnan(X)] = 0
    y_pred = model.predict_proba(X)[:, 1]
    churn_descision = (y_pred >= treshold).astype(int)
    df_original['churn_predicted'] = churn_descision
    df_original['churn_predicted_probability'] = y_pred

    return df_original

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# Sidebar section start
with st.sidebar:
    st.title('🗂 Ввод данных')
    
    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])
    with tab1:
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01)
            prediction_button = st.button('Предсказать', type='primary', use_container_width=True)
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
                
            
            

# Sidebar section end

# Main section start
st.image('https://miro.medium.com/v2/resize:fit:1400/1*WqId29D5dN_8DhiYQcHa2w.png', width=400)
st.title('Прогнозирование оттока клиентов')

with st.expander("Описание проекта"):
    st.write("""
    В данном проекте мы рассмотрим задачу прогнозирования оттока клиентов.
    Для этого мы будем использовать датасет из открытых источников.
    Датасет содержит информацию о клиентах, которые уже ушли или остались в компании.
    Наша задача - построить модель, которая будет предсказывать отток клиентов.
    """)

if len(st.session_state['df_input']) > 0:
    st.subheader('Данные из файла')
    st.write(st.session_state['df_input'])

if len(st.session_state['df_predicted']) > 0:
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-churn-predicted-all.csv',
        mime='text/csv',
    )