import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import os

st.set_page_config(page_title="مشروع المهندس أحمد رافد", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏠 نظام التنبؤ العقاري المتقدم (LSTM vs GRU vs RF)</h1>", unsafe_allow_html=True)

@st.cache_resource
def train_models():
    if not os.path.exists('final_cleaned_train.csv'): return None
    df = pd.read_csv('final_cleaned_train.csv')
    
    # تحويل النصوص إلى أرقام وحفظ القوائم
    cat_cols = df.select_dtypes(['object']).columns
    mappings = {col: list(df[col].astype('category').cat.categories) for col in cat_cols}
    for col in cat_cols: df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    
    # تهيئة البيانات للشبكات العصبية
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_scaled = sx.fit_transform(X)
    y_scaled = sy.fit_transform(y.reshape(-1, 1))
    # تحويل البيانات إلى 3D لتناسب LSTM و GRU
    X_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    def get_m(y_t, y_p):
        return mean_absolute_error(y_t, y_p), np.sqrt(mean_squared_error(y_t, y_p)), r2_score(y_t, y_p)

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100).fit(X, y)
    m_rf = get_m(y, rf.predict(X))

    # 2. LSTM
    lstm = Sequential([LSTM(50, activation='relu', input_shape=(1, X.shape[1])), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_3d, y_scaled, epochs=20, verbose=0)
    m_lstm = get_m(y, sy.inverse_transform(lstm.predict(X_3d)))

    # 3. GRU
    gru = Sequential([GRU(50, activation='relu', input_shape=(1, X.shape[1])), Dense(1)])
    gru.compile(optimizer='adam', loss='mse')
    gru.fit(X_3d, y_scaled, epochs=20, verbose=0)
    m_gru = get_m(y, sy.inverse_transform(gru.predict(X_3d)))

    return rf, lstm, gru, X.columns.tolist(), sx, sy, m_rf, m_lstm, m_gru, mappings

data = train_models()
if data:
    rf, lstm, gru, features, sx, sy, m_rf, m_lstm, m_gru, mappings = data
    st.sidebar.header("📝 إدخال المواصفات")
    u_in = {}
    for f in features:
        if f in mappings:
            u_in[f] = mappings[f].index(st.sidebar.selectbox(f, mappings[f]))
        else:
            u_in[f] = st.sidebar.number_input(f, value=float(0))

    if st.sidebar.button("🚀 بدء التوقع والمقارنة"):
        in_df = pd.DataFrame([u_in])[features]
        in_s = sx.transform(in_df)
        in_3d = in_s.reshape((1, 1, in_s.shape[1]))

        res_rf = rf.predict(in_df)[0]
        res_lstm = sy.inverse_transform(lstm.predict(in_3d))[0][0]
        res_gru = sy.inverse_transform(gru.predict(in_3d))[0][0]

        st.subheader("📊 جدول المقاييس والمقارنة")
        st.table(pd.DataFrame({
            "الموديل": ["Random Forest", "LSTM", "GRU"],
            "MAE": [f"{m_rf[0]:,.2f}", f"{m_lstm[0]:,.2f}", f"{m_gru[0]:,.2f}"],
            "RMSE": [f"{m_rf[1]:,.2f}", f"{m_lstm[1]:,.2f}", f"{m_gru[1]:,.2f}"],
            "R² Score": [f"{m_rf[2]:.4f}", f"{m_lstm[2]:.4f}", f"{m_gru[2]:.4f}"]
        }))
        st.success(f"توقع RF: ${res_rf:,.2f} | توقع LSTM: ${res_lstm:,.2f} | توقع GRU: ${res_gru:,.2f}")
