import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import os

# إعداد الصفحة
st.set_page_config(page_title="مشروع المهندس أحمد رافد", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏠 نظام التنبؤ العقاري المتقدم (LSTM vs GRU vs RF)</h1>", unsafe_allow_html=True)

@st.cache_resource
def train_models():
    if not os.path.exists('final_cleaned_train.csv'): 
        return None
    
    df = pd.read_csv('final_cleaned_train.csv')
    
    # 1. معالجة البيانات
    cat_cols = df.select_dtypes(['object']).columns
    mappings = {col: list(df[col].astype('category').cat.categories) for col in cat_cols}
    for col in cat_cols: 
        df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    
    # 2. تقسيم البيانات (70% تدريب، 15% تحقق، 15% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # 3. تهيئة المقاييس
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = sx.fit_transform(X_train)
    y_train_scaled = sy.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = sx.transform(X_test)
    
    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    def get_metrics(y_true, y_pred):
        return (mean_absolute_error(y_true, y_pred), 
                np.sqrt(mean_squared_error(y_true, y_pred)), 
                r2_score(y_true, y_pred))

    # --- تدريب النماذج ---
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    m_rf = get_metrics(y_test, rf.predict(X_test))

    # 2. LSTM
    lstm = Sequential([LSTM(32, activation='relu', input_shape=(1, X.shape[1])), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_train_3d, y_train_scaled, epochs=10, validation_split=0.17, verbose=0)
    m_lstm = get_metrics(y_test, sy.inverse_transform(lstm.
