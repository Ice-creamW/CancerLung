import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Lung Cancer Risk Predictor", layout="wide")

st.title("🫁 Lung Cancer Risk Prediction")
st.write("แอปพลิเคชันสำหรับทำนายระดับความเสี่ยงโรคมะเร็งปอดโดยใช้ Random Forest")

# --- 2. การจัดการข้อมูล ---
# โหลดข้อมูล (ในที่นี้ใช้ไฟล์ที่คุณอัปโหลดมา)
@st.cache_data
def load_data():
    df = pd.read_csv('LCR 6 features.xlsx.csv')
    return df

df = load_data()

# เตรียมข้อมูลสำหรับ Training
# ลบคอลัมน์ที่ไม่จำเป็นออก (Index และ Patient Id)
X = df.drop(['index', 'Patient Id', 'Level of risk'], axis=1)
y = df['Level of risk']

# แปลงเป้าหมาย (y) ให้เป็นตัวเลข (Low=0, Medium=1, High=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# สร้างและฝึกโมเดล Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# --- 3. ส่วนของ UI สำหรับผู้ใช้งาน (Sidebar) ---
st.sidebar.header("กรอกข้อมูลผู้ป่วย")

def user_input_features():
    age = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), 30)
    gender = st.sidebar.selectbox("Gender (1: Male, 2: Female)", (1, 2))
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
st.subheader("ข้อมูลที่รับเข้ามา")
st.write(input_df)

if st.button("ทำนายผล"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    res_label = le.inverse_transform(prediction)[0]
    
    st.subheader("ผลการทำนาย")
    # แสดงสีตามระดับความเสี่ยง
    if res_label == 'High':
        st.error(f"ระดับความเสี่ยง: {res_label}")
    elif res_label == 'Medium':
        st.warning(f"ระดับความเสี่ยง: {res_label}")
    else:
        st.success(f"ระดับความเสี่ยง: {res_label}")
        
    st.subheader("ความน่าจะเป็น (%)")
    prob_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    st.bar_chart(prob_df.T)