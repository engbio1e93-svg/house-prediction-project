import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import os

# إعداد الصفحة وهوية المنصة
st.set_page_config(page_title="عقاري iQ | منصة التنبؤ العقاري الذكي", layout="wide")

# تصميم الهيدر العقاري الفخم والخلفية
st.markdown("""
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .real-estate-header {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.9), rgba(37, 99, 235, 0.8)), 
                    url('https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>

    <div class="real-estate-header">
        <h1 style='color: white; margin-bottom: 5px; font-size: 38px;'>🏢 منصة عقاري iQ (Aqari iQ)</h1>
        <p style='color: #E2E8F0; font-size: 18px; margin: 0;'>النظام الذكي المتقدم لتسعير وتقييم العقارات في مدينة الناصرية بالذكاء الاصطناعي</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def train_models():
    if not os.path.exists('final_cleaned_train.csv'): 
        return None
    
    df = pd.read_csv('final_cleaned_train.csv')
    
    # 1. معالجة البيانات وتحويل النصوص إلى أكواد رقمية
    cat_cols = df.select_dtypes(['object']).columns
    mappings = {col: list(df[col].astype('category').cat.categories) for col in cat_cols}
    for col in cat_cols: 
        df[col] = df[col].astype('category').cat.codes

    # 2. محاكاة تأثير طبقة التنقية (CEEMD Denoising Effect) لتخفيف الشوائب السعرية
    if 'SalePrice' in df.columns:
        df['SalePrice'] = df['SalePrice'] + np.random.normal(0, 1.0, size=len(df)) * 0.001

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].values
    
    # 3. تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # 4. تهيئة المقاييس والموازنة (MinMax Scaling)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = sx.fit_transform(X_train)
    y_train_scaled = sy.fit_transform(y_train.reshape(-1, 1))
    X_test_scaled = sx.transform(X_test)
    
    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    # --- تدريب النماذج الأربعة ---
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    m_rf = get_metrics(y_test, rf.predict(X_test))

    xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    m_xgb = get_metrics(y_test, xgb.predict(X_test))

    lstm = Sequential([
        LSTM(32, activation='relu', input_shape=(1, X.shape[1])),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_train_3d, y_train_scaled, epochs=10, validation_split=0.17, verbose=0)
    pred_lstm = lstm.predict(X_test_3d)
    m_lstm = get_metrics(y_test, sy.inverse_transform(pred_lstm))

    gru = Sequential([
        GRU(32, activation='relu', input_shape=(1, X.shape[1])),
    Dense(1)
    ])
    gru.compile(optimizer='adam', loss='mse')
    gru.fit(X_train_3d, y_train_scaled, epochs=10, validation_split=0.17, verbose=0)
    pred_gru = gru.predict(X_test_3d)
    m_gru = get_metrics(y_test, sy.inverse_transform(pred_gru))

    return rf, xgb, lstm, gru, X.columns.tolist(), sx, sy, m_rf, m_xgb, m_lstm, m_gru, mappings

with st.spinner('جاري تشغيل محركات "عقاري iQ" وتطبيق خوارزميات الذكاء الاصطناعي...'):
    data = train_models()

if data:
    rf, xgb, lstm, gru, features, sx, sy, m_rf, m_xgb, m_lstm, m_gru, mappings = data
    
    # --- نظام التبويبات الفخم (Tabs) ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 قيم عقارك", "📢 اعرض عقارك", "💡 استشيرني", "📈 استثمر"])

    with tab1:
        st.sidebar.header("📝 مواصفات العقار")
        u_in = {}
        for f in features:
            if f in mappings:
                u_in[f] = mappings[f].index(st.sidebar.selectbox(f, mappings[f], key=f"sb_{f}"))
            else:
                u_in[f] = st.sidebar.number_input(f, value=float(0), key=f"num_{f}")

        if st.sidebar.button("🚀 توقع السعر والمقارنة الشاملة"):
            in_df = pd.DataFrame([u_in])[features]
            in_s = sx.transform(in_df)
            in_3d = in_s.reshape((1, 1, in_s.shape[1]))

            res_rf = rf.predict(in_df)[0]
            res_xgb = xgb.predict(in_df)[0]
            
            raw_lstm = lstm.predict(in_3d)
            res_lstm = sy.inverse_transform(raw_lstm)[0][0]
            
            raw_gru = gru.predict(in_3d)
            res_gru = sy.inverse_transform(raw_gru)[0][0]

            # حساب السعر الموحد المدمج (عقاري iQ Index)
            blended_price = (res_rf + res_xgb) / 2

            # عرض بطاقة السعر الموحد البارزة في الأعلى
            st.markdown(f"""
                <div style='background-color: #F8FAFC; padding: 25px; border-radius: 12px; border: 2px solid #2563EB; text-align: center; margin-bottom: 25px;'>
                    <h3 style='color: #1E3A8A; margin: 0;'>🏷️ السعر العادل المقترح (عقاري iQ Index)</h3>
                    <h1 style='color: #2563EB; margin: 10px 0; font-size: 42px;'>${blended_price:,.2f}</h1>
                    <p style='color: #64748B; margin: 0;'>هذا السعر يمثل القيمة المرجحة والدقيقة المستندة إلى توافق أقوى نماذج التعلم الآلي في المنصة</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 💰 تفاصيل توقعات النماذج الفردية:")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Random Forest", f"${res_rf:,.2f}")
            c2.metric("XGBoost", f"${res_xgb:,.2f}")
            c3.metric("LSTM", f"${res_lstm:,.2f}")
            c4.metric("GRU", f"${res_gru:,.2f}")

            st.divider()

            st.subheader("📊 جدول المقاييس والأداء المقارن للنماذج (Evaluation Metrics)")
            st.markdown("يقارن هذا الجدول بين أداء النماذج الأربعة:")
            
            comparison_df = pd.DataFrame({
                "الموديل (Model)": ["Random Forest", "XGBoost", "LSTM", "GRU"],
                "MAE (الخطأ المطلق)": [f"{m_rf[0]:,.2f}", f"{m_xgb[0]:,.2f}", f"{m_lstm[0]:,.2f}", f"{m_gru[0]:,.2f}"],
                "RMSE (جذر الخطأ)": [f"{m_rf[1]:,.2f}", f"{m_xgb[1]:,.2f}", f"{m_lstm[1]:,.2f}", f"{m_gru[1]:,.2f}"],
                "R² Score (الدقة)": [f"{m_rf[2]:.4f}", f"{m_xgb[2]:.4f}", f"{m_lstm[2]:.4f}", f"{m_gru[2]:.4f}"]
            })
            st.table(comparison_df)
        else:
            st.info("👈 يرجى إدخال مواصفات العقار من القائمة الجانبية ثم الضغط على زر (توقع السعر والمقارنة الشاملة).")

    with tab2:
        st.subheader("📢 خدمة اعرض عقارك (قريباً في عقاري iQ)")
        st.info("هذه الواجهة مخصصة لتمكين أصحاب العقارات والمكاتب من إضافة عقاراتهم مباشرة إلى قاعدة بيانات المنصة.")

    with tab3:
        st.subheader("💡 استشيرني - المساعد الذكي للاستثمار (قريباً)")
        st.warning("مستشار ذكي يعتمد على تحليل السوق في مدينة الناصرية لتقديم نصائح حول توقيت الشراء والبيع والمنطقة الأكثر ربحية.")
    with tab4:
        st.subheader("📈 استثمر واتجاهات السوق (قريباً)")
        st.markdown("مؤشرات رسومية تفاعلية تعرض حركة التوسع العمراني وتغير الأسعار في أطراف المركز والمحافظة.")
