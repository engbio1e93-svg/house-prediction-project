import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

# 1. الإعدادات والجمالية
st.set_page_config(page_title="نظام أحمد رافد", layout="wide")
st.markdown("<h1 style='text-align:center; color:#1e3d59;'>🏠 نظام التنبؤ العقاري - أحمد رافد</h1>", unsafe_allow_html=True)
st.write("---")

# 2. تدريب الموديل (مختصر)
@st.cache_resource
def train():
    if not os.path.exists('final_cleaned_train.csv'): return None
    df = pd.read_csv('final_cleaned_train.csv')
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    maps = {}
    for c in X.select_dtypes('object'):
        X[c] = X[c].astype('category')
        maps[c] = dict(enumerate(X[c].cat.categories))
        X[c] = X[c].cat.codes
    return {'m': RandomForestRegressor(100, random_state=42).fit(X, y), 'f': X.columns.tolist(), 'p': maps}

b = train()

# 3. الواجهة والتنبؤ
if b:
    labels = {'TotalArea':'مساحة الأرض','LivingArea':'المساحة المعيشية','Neighborhood':'المنطقة','BuildingQuality':'الجودة (1-10)','Bedrooms':'الغرف','YearBuilt':'سنة البناء'}
    cl1, cl2 = st.columns(2)
    inps = {}
    for i, c in enumerate(b['f']):
        col = cl1 if i % 2 == 0 else cl2
        lbl = labels.get(c, c)
        if c in b['p']:
            ch = col.selectbox(f"📍 {lbl}", list(b['p'][c].values()))
            inps[c] = [k for k, v in b['p'][c].items() if v == ch][0]
        else:
            v_max = 15000 if 'Area' in c else 2030
            inps[c] = col.number_input(f"🔢 {lbl}", 0, v_max, 10 if 'Year' in c else 1)

    if st.button("✨ حساب السعر المتوقع الآن"):
        res = b['m'].predict(pd.DataFrame([inps])[b['f']])
        st.balloons()
        st.success(f"💰 السعر المتوقع بواسطة نموذج أحمد رافد هو: ${res[0]:,.2f}")
else:
    st.error("تأكد من وجود ملف final_cleaned_train.csv")

st.markdown("<p style='text-align:center; color:grey;'>تصميم أحمد رافد © 2026</p>", unsafe_allow_html=True)