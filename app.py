import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")

st.title("🫁 Lung Cancer Severity Assessment")
st.write("แอปพลิเคชันพยากรณ์ความเสี่ยงโรคมะเร็งปอด")

# --- 2. การจัดการข้อมูล ---
@st.cache_data
def load_data():
    # ใช้ชื่อไฟล์ตามที่คุณอัปโหลดเป๊ะๆ
    file_name = 'LCR 6 features.xlsx.csv'
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        return df
    else:
        st.error(f"❌ ไม่พบไฟล์ '{file_name}' ใน GitHub")
        st.info("คำแนะนำ: ตรวจสอบใน GitHub ว่าชื่อไฟล์สะกดตรงกันทุกตัวอักษร (รวมถึงช่องว่างและจุด)")
        return None

df = load_data()

if df is not None:
    # --- เตรียมข้อมูลสำหรับ Training ---
    # ลบคอลัมน์ที่ไม่เกี่ยวข้อง รวมถึง Age และ Gender ตามที่ต้องการ
    cols_to_drop = ['index', 'Patient Id', 'Age', 'Gender', 'Level of risk']
    X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    y = df['Level of risk']

    # แปลงเป้าหมาย (y) ให้เป็นตัวเลข
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # สร้างและฝึกโมเดล (ใช้เฉพาะฟีเจอร์ที่เหลืออยู่)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    # --- 3. ส่วนของ UI สำหรับผู้ใช้งาน (Sidebar) ---
    st.sidebar.header("กรอกข้อมูลอาการและพฤติกรรม")

    def user_input_features():
        # ลบ Slider ของ Age และ Selectbox ของ Gender ออกแล้ว
        alcohol = st.sidebar.slider("Alcohol use (การดื่มสุรา)", 1, 8, 4)
        smoking = st.sidebar.slider("Smoking (การสูบบุหรี่)", 1, 8, 3)
        passive_smoker = st.sidebar.slider("Passive Smoker (รับควันบุหรี่มือสอง)", 1, 8, 2)
        cough_blood = st.sidebar.slider("Coughing of Blood (ไอเป็นเลือด)", 1, 9, 4)
        fatigue = st.sidebar.slider("Fatigue (ความเหนื่อยล้า)", 1, 9, 3)
        wheezing = st.sidebar.slider("Wheezing (เสียงหายใจวี้ด)", 1, 8, 2)
        
        # ส่งข้อมูลเฉพาะฟีเจอร์ที่ใช้ทำนาย
        data = {
            'Alcohol use': alcohol,
            'Smoking': smoking,
            'Passive Smoker': passive_smoker,
            'Coughing of Blood': cough_blood,
            'Fatigue': fatigue,
            'Wheezing': wheezing
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # --- 4. การแสดงผลการทำนาย ---
    st.subheader("📋 ข้อมูลอาการที่ระบุ")
    st.write(input_df)

    if st.button("Prediction"):
        # ทำนายผลโดยใช้ฟีเจอร์ที่ไม่มี Age และ Gender
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        res_label = le.inverse_transform(prediction)[0]
        
        st.subheader("🎯 ผลการพยากรณ์")
        if res_label == 'High‼':
            st.error(f"ระดับความเสี่ยง: {res_label} (สูง)")
        elif res_label == 'Medium':
            st.warning(f"ระดับความเสี่ยง: {res_label} (ปานกลาง)")
        else:
            st.success(f"ระดับความเสี่ยง: {res_label} ")
            
    
