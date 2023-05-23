import streamlit as st
import pickle
import pandas as pd



def add_sidebar():
  st.sidebar.header("Measurements")
    
  slider_labels = [
    ("Age","Age of the person","age","18", "90", 0),
    ("Gender","Anonymized gender of the person","sex","0", "0", 1),
    ("Chest Pain","Chest Pain type chest pain type Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic","cp","0", "3", 0),
    ("Blood Pressure","resting blood pressure (in mm Hg)","trtbps","94", "200", 0),
    ("Cholestoral","cholestoral in mg/dl fetched via BMI sensor","chol","126", "564", 0),
    ("Blood sugar","(is fasting blood sugar > 120 mg/dl)","fbs","0", "0", 1),
    ("ECG","resting electrocardiographic results Value 0: normal, Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria","restecg","0", "2", 0),
    ("Max HR","maximum heart rate achieved","thalachh","40", "250", 0),
    ("Exercise Angina","Is patient experiencing exercise induced angina?","exng","0", "0", 1),
    ("Old Peak","Previous peak","oldpeak","0", "6.2", 0),
    ("Slope","Slope","slp","0", "2", 0),
    ("CAA","number of major vessels (0-3)","caa","0", "3", 0)
  ]

  input_dict = {}

  for label, desc, key, min, max, switch in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      help=desc,
      min_value=float(min),
      max_value=float(max),
      value=float(max)/2,
    ) if switch == 0 else st.sidebar.checkbox(label, value=False, help=desc)
    
  return input_dict


def main():
  st.set_page_config(
    page_title="Project: Cardionaut - Heart Attack AI predictor",
    page_icon=":broken_heart:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()
  
  with st.container():
    st.title("Project: Cardionaut")
    st.write("Heart Attack AI predictor")
  
  col1, col2 = st.columns([4,1])

  st.write(input_data)
  


 
if __name__ == '__main__':
  main()