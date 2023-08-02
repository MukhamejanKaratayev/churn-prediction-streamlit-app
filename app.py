import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px


# Настройка страницы
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальные переменные
uploaded_file = None

# Создание переменных session state
if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
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

# Кэширование функции предсказания
@st.cache_data
def predict_churn(df_input, treshold):
    # Функция для предсказания оттока
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
    # Функция для конвертации датафрейма в csv
    return df.to_csv(index=False).encode('utf-8')

# Sidebar section start
# Сайдбар блок
with st.sidebar:
    st.title('🗂 Ввод данных')
    
    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])
    with tab1:
        # Вкладка с загрузкой файла, выбором порога и кнопкой предсказания (вкладка 1)
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv', 'xlsx'], on_change=reset_session_state)
        if uploaded_file is not None:
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01, key='slider1')
            prediction_button = st.button('Предсказать', type='primary', use_container_width=True, key='button1')
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
                st.session_state['tab_selected'] = 'tab1'

    with tab2:
        # Вкладка с вводом данных вручную, выбором порога и кнопкой предсказания (вкладка 2)
        customer_id = st.text_input('Customer ID', placeholder='0000', help='Введите ID клиента')
        gender = st.selectbox( 'Пол', ('female', 'male'))
        senior_citizen = st.selectbox('Пенсионер', ('Да', 'Нет'))
        partner = st.selectbox('Партнер', ('yes', 'no'))
        dependents = st.selectbox('Иждивенцы', ('yes', 'no'))
        tenure = st.number_input('Количество месяцев, в течение которых клиент остается с компанией', min_value=0, max_value=100, value=0)
        phone_service = st.selectbox('Телефон', ('yes', 'no'))
        multiple_lines = st.selectbox('Несколько линий', ('yes', 'no', 'no_phone_service'))
        internet_service = st.selectbox('Интернет', ('dsl', 'fiber_optic', 'no'))
        online_security = st.selectbox('Онлайн безопасность', ('yes', 'no', 'no_internet_service'))
        online_backup = st.selectbox('Онлайн резервное копирование', ('yes', 'no', 'no_internet_service'))
        device_protection = st.selectbox('Защита устройства', ('yes', 'no', 'no_internet_service'))
        tech_support = st.selectbox('Техническая поддержка', ('yes', 'no', 'no_internet_service'))
        streaming_tv = st.selectbox('Стриминговое ТВ', ('yes', 'no', 'no_internet_service'))
        streaming_movies = st.selectbox('Стриминговые фильмы', ('yes', 'no', 'no_internet_service'))
        contract = st.selectbox('Контракт', ('month-to-month', 'one_year', 'two_year'))
        paperless_billing = st.selectbox('Бумажный чек', ('yes', 'no'))
        payment_method = st.selectbox('Способ оплаты', ('bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'))
        monthly_charges = st.number_input('Ежемесячные платежи', min_value=0, max_value=1000, value=0)
        total_charges = st.number_input('Общие платежи', min_value=0, max_value=100000, value=0)        
        
        # Если введен ID клиента, то показываем слайдер с порогом и кнопку предсказания
        if customer_id != '':
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01, key='slider2')
            prediction_button_tab2 = st.button('Предсказать', type='primary', use_container_width=True, key='button2')
            
            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'
                # Сохраняем введенные данные в session state в виде датафрейма
                st.session_state['df_input'] = pd.DataFrame({
                    'customerid': customer_id,
                    'gender': gender,
                    'seniorcitizen': 1 if senior_citizen == 'Да' else 0,
                    'partner': partner,
                    'dependents': dependents,
                    'tenure': int(tenure),
                    'phoneservice': phone_service,       
                    'multiplelines': multiple_lines,
                    'internetservice': internet_service,
                    'onlinesecurity': online_security,
                    'onlinebackup': online_backup,
                    'deviceprotection': device_protection,
                    'techsupport': tech_support,
                    'streamingtv': streaming_tv,
                    'streamingmovies': streaming_movies,
                    'contract': contract,
                    'paperlessbilling' : paperless_billing,
                    'paymentmethod': payment_method,
                    'monthlycharges': monthly_charges,
                    'totalcharges': total_charges 
                }, index=[0])
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)

                

# Sidebar section end

# Main section start
# Основной блок
st.image('https://miro.medium.com/v2/resize:fit:1400/1*WqId29D5dN_8DhiYQcHa2w.png', width=400)
st.title('Прогнозирование оттока клиентов')

with st.expander("Описание проекта"):
    st.write("""
    В данном проекте мы рассмотрим задачу прогнозирования оттока клиентов.
    Для этого мы будем использовать датасет из открытых источников.
    Датасет содержит информацию о клиентах, которые уже ушли или остались в компании.
    Наша задача - построить модель, которая будет предсказывать отток клиентов.
    """)

# Вывод входных данных (из файла или введенных пользователем)
if len(st.session_state['df_input']) > 0:
    # Если предсказание еще не было сделано, то выводим входные данные в общем виде
    if len(st.session_state['df_predicted']) == 0:
        st.subheader('Данные из файла')
        st.write(st.session_state['df_input'])
    else:
        # Если предсказание уже было сделано, то выводим входные данные в expander
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])
    # Примеры визуализации данных
    # st.line_chart(st.session_state['df_input'][['tenure', 'monthlycharges']])
    # st.bar_chart(st.session_state['df_input'][['contract']])


# Выводим результаты предсказания для отдельного клиента (вкладка 2)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
    if st.session_state['df_predicted']['churn_predicted'][0] == 0:
        st.image('https://gifdb.com/images/high/happy-face-steve-carell-the-office-057k667rwmncrjwh.gif', width=200)
        st.subheader(f'Клиент :green[остается] c вероятностью {(1 - st.session_state["df_predicted"]["churn_predicted_probability"][0]) * 100:.2f}%')
    else:
        st.image('https://media.tenor.com/QFnU4bhN8gMAAAAd/michael-scott-crying.gif', width=200)
        st.subheader(f'Клиент :red[уходит] c вероятностью {(st.session_state["df_predicted"]["churn_predicted_probability"][0]) * 100:.2f}%')


# Выводим результаты предсказания для клинтов из файла (вкладка 1)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab1':
    # Результаты предсказания для всех клиентов в файле
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    # Скачиваем результаты предсказания для всех клиентов в файле
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-churn-predicted-all.csv',
        mime='text/csv',
    )

    # Гистограмма оттока для всех клиентов в файле
    fig = px.histogram(st.session_state['df_predicted'], x='churn_predicted', color='churn_predicted')
    st.plotly_chart(fig, use_container_width=True)

    # Клиенты с высоким риском оттока
    risk_clients = st.session_state['df_predicted'][st.session_state['df_predicted']['churn_predicted'] == 1]
    # Выводим клиентов с высоким риском оттока
    if len(risk_clients) > 0:
        st.subheader('Клиенты с высоким риском оттока')
        st.write(risk_clients)
        # Скачиваем клиентов с высоким риском оттока
        res_risky_csv = convert_df(risk_clients)
        st.download_button(
            label="Скачать клиентов с высоким риском оттока",
            data=res_risky_csv,
            file_name='df-churn-predicted-risk-clients.csv',
            mime='text/csv',
        )