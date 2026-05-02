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
    
    # 1. معالجة البيانات وتحويل النصوص إلى أرقام
    cat_cols = df.select_dtypes(['object']).columns
    mappings = {col: list(df[col].astype('category').cat.categories) for col in cat_cols}
    for col in cat_cols: 
        df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    
    # 2. تقسيم البيانات (70% تدريب، 30% اختبار أولي ثم تقسيمها لاحقاً)
    # ملاحظة: سنستخدم 15% للاختبار النهائي و 15% للتحقق أثناء التدريب
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # 3. تهيئة المقاييس (Scaling)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = sx.fit_transform(X_train)
    y_train_scaled = sy.fit_transform(y_train.reshape(-1, 1))
    
    X_test_scaled = sx.transform(X_test)
    
    # تحويل البيانات إلى 3D لتناسب LSTM و GRU
    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    def get_metrics(y_true, y_pred):
        return (mean_absolute_error(y_true, y_pred), 
                np.sqrt(mean_squared_error(y_true, y_pred)), 
                r2_score(y_true, y_pred))

    # --- (1) Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    m_rf = get_metrics(y_test, rf.predict(X_test))

    # --- (2) LSTM مع Validation Split ---
    lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X.shape[1])),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    # استخدام 17% من بيانات التدريب للتحقق (ما يعادل 15% من الإجمالي)
    lstm.fit(X_train_3d, y_train_scaled, epochs=20, validation_split=0.17, verbose=0)
    m_lstm = get_metrics(y_test, sy.inverse_transform(lstm.predict(X_test_3d)))

    # --- (3) GRU مع Validation Split ---
    gru = Sequential([
        GRU(50, activation='relu', input_shape=(1, X.shape[1])),
        Dense(1)
    ])
    gru.compile(optimizer='adam', loss='mse')
    gru.fit(X_train_3d, y_train_scaled, epochs=20, validation_split=0.17, verbose=0)
    m_gru = get_metrics(y_test, sy.inverse_transform(gru.predict(X_test_3d)))

    return rf, lstm, gru, X.columns.tolist(), sx, sy, m_rf, m_lstm, m_gru, mappings

# تشغيل التدريب
data = train_models()

if data:
    rf, lstm, gru, features, sx, sy, m_rf, m_lstm, m_gru, mappings = data
    
    # واجهة المستخدم الجانبية
    st.sidebar.header("📝 إدخال مواصفات العقار")
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

        # التنبؤات
        res_rf = rf.predict(in_df)[0]
        res_lstm = sy.inverse_transform(lstm.predict(in_3d))[0][0]
        res_gru = sy.inverse_transform(gru.predict(in_3d))[0][0]

        # عرض النتائج في جدول المقارنة
        st.subheader("📊 تقييم أداء النماذج (على بيانات الاختبار)")
        results_df = pd.DataFrame({
            "الموديل (Model)": ["Random Forest", "LSTM", "GRU"],
            "MAE (متوسط الخطأ)": [f"{m_rf[0]:,.2f}", f"{m_lstm[0]:,.2f}", f"{m_gru[0]:,.2f}"],
            "RMSE (جذر الخطأ)": [f"{m_rf[1]:,.2f}", f"{m_lstm[1]:,.2f}", f"{m_gru[1]:,.2f}"],
            "R² Score (الدقة)": [f"{m_rf[2]:.4f}", f"{m_lstm[2]:.4f}", f"{m_gru[2]:.4f}"]
        })
        st.table(results_df)

        # عرض الأسعار المتوقعة
        col1, col2, col3 = st.columns(3)
        col1.metric("توقع RF", f"${res_rf:,.2f}")
        col2.metric("توقع LSTM", f"${res_lstm:,.2f}")
        col3.metric("توقع GRU", f"${res_gru:,.2f}")
        
        st.success("تمت عملية التنبؤ بنجاح بناءً على نماذج الاختبار والتحقق.")
else:
    st.error("لم يتم العثور على ملف البيانات 'final_cleaned_train.csv'. يرجى التأكد من وجوده في مجلد المشروع.")
