import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")

st.title("🫁 Lung Cancer Risk Prediction")
st.write("แอปพลิเคชันพยากรณ์ความเสี่ยงโรคมะเร็งปอด (Random Forest Model)")

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
    # เตรียมข้อมูลสำหรับ Training
    # ลบคอลัมน์ที่ไม่เกี่ยวข้องกับการพยากรณ์
    cols_to_drop = ['index', 'Patient Id', 'Level of risk']
    X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    y = df['Level of risk']

    # แปลงเป้าหมาย (y) ให้เป็นตัวเลข
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # สร้างและฝึกโมเดล
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    # --- 3. ส่วนของ UI สำหรับผู้ใช้งาน (Sidebar) ---
    st.sidebar.header("กรอกข้อมูลผู้ป่วย")

    def user_input_features():
        # ดึงค่า Min/Max จากข้อมูลจริงมาสร้าง Slider
        age = st.sidebar.slider("Age (อายุ)", int(df.Age.min()), int(df.Age.max()), 30)
        gender = st.sidebar.selectbox("Gender (เพศ - 1: ชาย, 2: หญิง)", (1, 2))
        alcohol = st.sidebar.slider("Alcohol use", 1, 8, 4)
        smoking = st.sidebar.slider("Smoking", 1, 8, 3)
        passive_smoker = st.sidebar.slider("Passive Smoker", 1, 8, 2)
        cough_blood = st.sidebar.slider("Coughing of Blood", 1, 9, 4)
        fatigue = st.sidebar.slider("Fatigue", 1, 9, 3)
        wheezing = st.sidebar.slider("Wheezing", 1, 8, 2)
        
        data = {
            'Age': age,
            'Gender': gender,
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
    st.subheader("📋 ข้อมูลที่คุณกรอก")
    st.write(input_df)

    if st.button("ทำการพยากรณ์"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        res_label = le.inverse_transform(prediction)[0]
        
        st.subheader("🎯 ผลการพยากรณ์")
        if res_label == 'High':
            st.error(f"ระดับความเสี่ยง: {res_label}")
        elif res_label == 'Medium':
            st.warning(f"ระดับความเสี่ยง: {res_label}")
        else:
            st.success(f"ระดับความเสี่ยง: {res_label}")
            
        # แสดงกราฟความน่าจะเป็น
        st.subheader("📊 ความน่าจะเป็นในแต่ละระดับ (%)")
        prob_df = pd.DataFrame(prediction_proba, columns=le.classes_)
        st.bar_chart(prob_df.T)
