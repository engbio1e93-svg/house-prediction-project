import os
import uuid
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

st.set_page_config(page_title="عقاري iQ | منصة التنبؤ العقاري الذكي", page_icon="🏡", layout="wide")

# الهوية البصرية الخضراء
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Cairo',sans-serif;}
.stApp {background:#f5f8f5;}
section[data-testid="stSidebar"] {background:#103b2e;}
section[data-testid="stSidebar"] * {color:#fff;}
.hero {background:linear-gradient(120deg,rgba(8,43,32,.96),rgba(31,103,72,.86)),
url('https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=1800&q=85');
background-position:center;background-size:cover;padding:36px 20px;border-radius:18px;
color:white;text-align:center;margin-bottom:22px;}
.hero h1 {color:white;font-size:36px;margin:0 0 8px;}
.hero p {color:#e2f2e8;margin:0;}
div[data-testid="stMetric"] {background:white;border:1px solid #e1ebe4;padding:14px;border-radius:13px;}
.stButton>button,.stFormSubmitButton>button {background:#176348;color:white;border:0;border-radius:10px;font-weight:700;}
.stButton>button:hover,.stFormSubmitButton>button:hover {background:#104b36;color:white;}
</style>
<div class="hero"><h1>🏡 منصة عقاري iQ</h1>
<p>النظام الذكي لتقييم العقارات واستعراض الفرص العقارية في مدينة الناصرية</p></div>
""", unsafe_allow_html=True)

# الاتصال بقاعدة Supabase من إعدادات Secrets
@st.cache_resource
def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None

supabase = get_supabase()
BUCKET = "listing-images"

def fetch_listings():
    if supabase is None:
        return []
    try:
        return (supabase.table("listings").select(
            "id,owner_name,phone_number,address,area,bedrooms,floors,bathrooms,"
            "build_quality,image,listing_type,price"
        ).order("id", desc=True).execute().data or [])
    except Exception as e:
        st.error(f"تعذر جلب الإعلانات: {e}")
        return []

def upload_image(file):
    if not file:
        return None
    ext = file.name.rsplit(".", 1)[-1].lower()
    path = f"{uuid.uuid4().hex}.{ext}"
    supabase.storage.from_(BUCKET).upload(
        path, file.getvalue(),
        {"content-type": file.type, "upsert": "false"}
    )
    return supabase.storage.from_(BUCKET).get_public_url(path)

def insert_listing(payload):
    return supabase.table("listings").insert(payload).execute()

shamia_areas = ["شارع بغداد","الشموخ","الشعلة","المنصورية","الثورة","حي المنتزه",
"الزاوية","الإسكان القديم","حي المختار","السكك","الإسكان الصناعي","حي الرسول"]
jazeera_areas = ["سومر","حي الشهداء","الفداء","حي الحسين","حي أور","شارع السبيس",
"الموحية","المدينة","أريدو","الإدارة المحلية","الحبوبي","السراي"]

# تدريب نماذج التقييم من ملف البيانات فقط، دون استخدام إعلانات Supabase
@st.cache_resource(show_spinner=False)
def train_models():
    if not os.path.exists("final_cleaned_train.csv"):
        return None, "ملف final_cleaned_train.csv غير موجود بجانب app.py."
    try:
        df = pd.read_csv("final_cleaned_train.csv")
        if "SalePrice" not in df.columns:
            return None, "عمود SalePrice غير موجود في ملف البيانات."
        df = df.dropna(subset=["SalePrice"]).copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        mappings = {}
        for col in cat_cols:
            series = df[col].fillna("غير محدد").astype("category")
            mappings[col] = list(series.cat.categories)
            df[col] = series.cat.codes
        X = df.drop(columns=["SalePrice"])
        y = pd.to_numeric(df["SalePrice"], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        sx, sy = MinMaxScaler(), MinMaxScaler()
        Xtr = sx.fit_transform(X_train)
        Xte = sx.transform(X_test)
        ytr = sy.fit_transform(y_train.reshape(-1, 1))
        Xtr3 = Xtr.reshape((Xtr.shape[0], 1, Xtr.shape[1]))
        Xte3 = Xte.reshape((Xte.shape[0], 1, Xte.shape[1]))

        def metric(actual, predicted):
            predicted = np.asarray(predicted).reshape(-1)
            return (mean_absolute_error(actual, predicted),
                    np.sqrt(mean_squared_error(actual, predicted)),
                    r2_score(actual, predicted))

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        mrf = metric(y_test, rf.predict(X_test))

        xgb = XGBRegressor(n_estimators=50, learning_rate=0.1,
                           random_state=42, objective="reg:squarederror")
        xgb.fit(X_train, y_train)
        mxgb = metric(y_test, xgb.predict(X_test))

        tf.random.set_seed(42)
        lstm = Sequential([LSTM(32, activation="tanh", input_shape=(1, X.shape[1])), Dense(1)])
        lstm.compile(optimizer="adam", loss="mse")
        lstm.fit(Xtr3, ytr, epochs=10, validation_split=0.17, verbose=0)
        mlstm = metric(y_test, sy.inverse_transform(lstm.predict(Xte3, verbose=0)))

        gru = Sequential([GRU(32, activation="tanh", input_shape=(1, X.shape[1])), Dense(1)])
        gru.compile(optimizer="adam", loss="mse")
        gru.fit(Xtr3, ytr, epochs=10, validation_split=0.17, verbose=0)
        mgru = metric(y_test, sy.inverse_transform(gru.predict(Xte3, verbose=0)))

        return (rf, xgb, lstm, gru, X.columns.tolist(), sx, sy,
                mrf, mxgb, mlstm, mgru, mappings), None
    except Exception as e:
        return None, f"حدث خطأ أثناء تدريب النماذج: {e}"

with st.spinner("جاري تشغيل نماذج الذكاء الاصطناعي..."):
    model_data, model_error = train_models()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 قيم عقارك", "🏘 العروض المتاحة", "📢 اعرض عقارك",
    "💡 استشيرني", "📈 استثمر", "📰 آخر الأخبار"
])

# 1. التقييم
with tab1:
    st.subheader("🏠 قيّم عقارك بالذكاء الاصطناعي")
    if model_data is None:
        st.error(model_error)
    else:
        rf, xgb, lstm, gru, features, sx, sy, mrf, mxgb, mlstm, mgru, mappings = model_data
        st.write("أدخل مواصفات العقار كما تظهر في بيانات النموذج.")
        with st.form("valuation_form"):
            values = {}
            c1, c2 = st.columns(2)
            for i, feature in enumerate(features):
                with (c1 if i % 2 == 0 else c2):
                    if feature in mappings:
                        opts = mappings[feature]
                        selected = st.selectbox(feature, opts, key=f"eval_{feature}")
                        values[feature] = opts.index(selected)
                    else:
                        values[feature] = st.number_input(feature, value=0.0, key=f"evaln_{feature}")
            evaluate = st.form_submit_button("🚀 توقع السعر")
        if evaluate:
            input_df = pd.DataFrame([values])[features]
            scaled = sx.transform(input_df)
            seq = scaled.reshape((1, 1, scaled.shape[1]))
            prf = float(rf.predict(input_df)[0])
            pxgb = float(xgb.predict(input_df)[0])
            plstm = float(sy.inverse_transform(lstm.predict(seq, verbose=0))[0][0])
            pgru = float(sy.inverse_transform(gru.predict(seq, verbose=0))[0][0])
            st.markdown(f"""<div style="background:#e8f3eb;padding:22px;border-radius:14px;text-align:center">
            <h3 style="color:#245b43">السعر العادل المقترح</h3>
            <h1 style="color:#176348">${(prf+pxgb)/2:,.2f}</h1>
            <p>متوسط تنبؤ Random Forest وXGBoost</p></div>""", unsafe_allow_html=True)
            cols = st.columns(4)
            for col, label, value in zip(cols, ["Random Forest","XGBoost","LSTM","GRU"], [prf,pxgb,plstm,pgru]):
                col.metric(label, f"${value:,.2f}")
            st.subheader("📊 مقاييس أداء النماذج")
            st.dataframe(pd.DataFrame({
                "النموذج":["Random Forest","XGBoost","LSTM","GRU"],
                "MAE":[mrf[0],mxgb[0],mlstm[0],mgru[0]],
                "RMSE":[mrf[1],mxgb[1],mlstm[1],mgru[1]],
                "R²":[mrf[2],mxgb[2],mlstm[2],mgru[2]]
            }), use_container_width=True, hide_index=True)
            st.caption("التقدير يعتمد على ملف التدريب الحالي ولا يمثل ضماناً لسعر البيع الفعلي.")

# 2. العروض: قراءة الأعمدة الموجودة فقط
with tab2:
    st.subheader("🏘 العروض العقارية في الناصرية")
    if supabase is None:
        st.warning("أضف إعدادات Supabase إلى Secrets حتى تظهر الإعلانات.")
    else:
        listings = fetch_listings()
        if not listings:
            st.info("لا توجد إعلانات محفوظة حالياً.")
        for item in listings:
            with st.container(border=True):
                if item.get("image"):
                    st.image(item["image"], use_container_width=True)
                st.markdown(f"### {item.get('listing_type') or 'عقار'}")
                price = item.get("price")
                if price is not None:
                    st.markdown(f"**السعر:** ${float(price):,.2f}")
                st.write(f"**العنوان:** {item.get('address') or 'غير محدد'}")
                st.write(f"**المنطقة:** {item.get('area') or 'غير محددة'}")
                st.write(f"**المساحة:** {item.get('area') or 'غير محددة'}")
                st.write(f"**صاحب الإعلان:** {item.get('owner_name') or 'غير محدد'}")
                st.write(f"**الهاتف:** {item.get('phone_number') or 'غير محدد'}")
                st.write(f"**غرف النوم:** {item.get('bedrooms') or 0} | **الطوابق:** {item.get('floors') or 0} | **الحمامات:** {item.get('bathrooms') or 0}")
                st.write(f"**جودة البناء:** {item.get('build_quality') or 'غير محددة'} / 10")

# 3. إضافة إعلان: لا يكتب إلا في الأعمدة الحالية
with tab3:
    st.subheader("📢 اعرض عقارك")
    st.write("سيتم حفظ بيانات الإعلان في جدول listings الحالي فقط.")
    if supabase is None:
        st.warning("الاتصال بـ Supabase غير جاهز. أضف SUPABASE_URL وSUPABASE_KEY في Secrets.")
    with st.form("listing_form", clear_on_submit=True):
        a, b = st.columns(2)
        with a:
            owner = st.text_input("اسم صاحب العقار / المكتب *")
            phone = st.text_input("رقم الهاتف *")
            listing_type = st.selectbox("نوع الإعلان", ["بيع", "إيجار"])
            side = st.selectbox("الصوب", ["صوب الشامية", "صوب الجزيرة"])
            areas = shamia_areas if side == "صوب الشامية" else jazeera_areas
            neighborhood = st.selectbox("المنطقة", areas)
        with b:
            address_detail = st.text_input("العنوان / أقرب نقطة دالة")
            total_area = st.number_input("المساحة بالمتر المربع", min_value=1.0, value=200.0)
            price = st.number_input("السعر بالدولار", min_value=1.0, value=75000.0)
            bedrooms = st.number_input("عدد غرف النوم", min_value=0, value=3)
            floors = st.number_input("عدد الطوابق", min_value=1, value=1)
        bathrooms = st.number_input("عدد الحمامات", min_value=0, value=2)
        quality = st.slider("جودة البناء من 1 إلى 10", 1, 10, 8)
        image_file = st.file_uploader("صورة واحدة للعقار (اختياري)", type=["jpg","jpeg","png"])
        publish = st.form_submit_button("🚀 نشر الإعلان")
    if publish:
        if not owner.strip() or not phone.strip():
            st.error("يرجى إدخال اسم صاحب العقار ورقم الهاتف.")
        elif supabase is None:
            st.error("لا يوجد اتصال بـ Supabase.")
        else:
            try:
                image_url = upload_image(image_file) if image_file else None
                payload = {
                    "owner_name": owner.strip(),
                    "phone_number": phone.strip(),
                    "address": f"{side} - {neighborhood} - {address_detail}".strip(" -"),
                    "area": float(total_area),
                    "bedrooms": int(bedrooms),
                    "floors": int(floors),
                    "bathrooms": int(bathrooms),
                    "build_quality": int(quality),
                    "image": image_url,
                    "listing_type": listing_type,
                    "price": float(price)
                }
                insert_listing(payload)
                st.success("تم نشر الإعلان وحفظه في Supabase بنجاح.")
                st.rerun()
            except Exception as e:
                st.error(f"فشل نشر الإعلان: {e}")

# 4. الاستشارة
with tab4:
    st.subheader("💡 استشيرني")
    st.info("قسم الاستشارة حالياً واجهة تعريفية، ويمكن ربطه لاحقاً بمساعد ذكي بعد تجهيز بيانات سوق محلية موثوقة.")
    st.write("سيكون بالإمكان تطوير توصيات للشراء والبيع واختيار المناطق بناءً على بيانات العقارات الفعلية.")

# 5. الاستثمار
with tab5:
    st.subheader("📈 حاسبة العائد الاستثماري")
    x, y = st.columns(2)
    with x:
        invest_price = st.number_input("سعر شراء العقار بالدولار", min_value=1.0, value=120000.0, key="inv_price")
    with y:
        rent = st.number_input("الإيجار الشهري بالدولار", min_value=0.0, value=550.0, key="monthly_rent")
    if st.button("احسب العائد السنوي"):
        roi = rent * 12 / invest_price * 100
        st.metric("العائد السنوي الإجمالي", f"{roi:.2f}%")
        st.caption("الحسبة لا تشمل الصيانة أو الضرائب أو فترات خلو العقار.")
        if roi >= 5.5:
            st.success("العائد أعلى من حد 5.5% المحدد في الحاسبة.")
        else:
            st.info("العائد أقل من 5.5%؛ قارن السعر والإيجار بعقارات مشابهة.")

# 6. الأخبار: نموذج توضيحي غير مباشر
with tab6:
    st.subheader("📰 أخبار ومستجدات القطاع العقاري")
    st.caption("المحتوى التالي توضيحي وليس خبراً مباشراً تم التحقق منه.")
    c1, c2 = st.columns([1,2])
    with c1:
        st.image("https://images.unsplash.com/photo-1541888946425-d0fbb186a5b3?auto=format&fit=crop&w=800&q=80",
                 caption="مشهد إنشائي توضيحي", use_container_width=True)
    with c2:
        st.markdown("### 🏗 التوسع العمراني والفرز العقاري")
        st.write("يمكن تخصيص هذا القسم لاحقاً لأخبار موثوقة عن مشاريع الإسكان والفرز والتطويب في محافظة ذي قار.")
        st.info("أضف مصدر الخبر وتاريخه قبل عرضه على أنه خبر حقيقي.")
