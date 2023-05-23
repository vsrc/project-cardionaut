import streamlit as st
import pickle
import pandas as pd
# import plotly.graph_objects as go
import numpy as np
from streamlit_toggle import st_toggle_switch



def add_sidebar():
  st.sidebar.header("Measurements")
    
  slider_labels = [
    ("Age","Age of the person","age","18", "90",  "half", 1, 0),
    ("Gender","Anonymized gender of the person","sex","0", "1",  "0", 1, 0),
    ("Chest Pain","Chest Pain type chest pain type Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic","cp","0", "3",  "half", 1, 0),
    ("Blood Pressure","resting blood pressure (in mm Hg)","trtbps","94", "200",  "half", 1, 0),
    ("Cholestoral","cholestoral in mg/dl fetched via BMI sensor","chol","126", "564",  "half", 1, 0),
    ("Blood sugar","(is fasting blood sugar > 120 mg/dl)","fbs","0", "0",  "half", 1, 1),
    ("ECG","resting electrocardiographic results Value 0: normal, Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria","restecg","0", "2",  "half", 1, 0),
    ("Max HR","maximum heart rate achieved","thalachh","40", "250",  "half", 1, 0),
    ("Exercise Angina","Is patient experiencing exercise induced angina?","exng","0", "0",  "half", 1, 1),
    ("Old Peak","Previous peak","oldpeak","0", "6.2",  "half", 0.01, 0),
    ("Slope","Slope","slp","0", "2",  "2", 1, 0),
    ("CAA","number of major vessels (0-3)","caa","0", "3",  "1", 1, 0)
  ]

  input_dict = {}

  for label, desc, key, min, max, default, step, switch in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      help=desc,
      step=float(step),
      min_value=float(min),
      max_value=float(max),
      value=float(max)/2 if default == "half" else float(default),
    ) if switch == 0 else st.sidebar.checkbox(label, value=False, help=desc)

  # st.write(slider_labels)
    
  return input_dict


def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)

  input_array_scaled = scaler.transform(input_array)

  prediction = model.predict(input_array_scaled)
  
  st.subheader("Heart Attack Prediction")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis low'>Low possibility!</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis high'>High possibility</span>", unsafe_allow_html=True)
    
  
  # st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  # st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
  
  # st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


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
    st.write("This is a simple AI model that predicts the probability of a heart attack based on a set of measurements.")
    st.write("The model is a simple Logistic Regression model with an accuracy of: 83.6%")
    

    add_predictions(input_data)
  


 
if __name__ == '__main__':
  main()