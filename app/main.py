import streamlit as st
import pickle
import pandas as pd



def main():
  st.set_page_config(
    page_title="Project: Cardionaut - Heart Attack AI predictor",
    page_icon=":broken_heart:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  # input_data = add_sidebar()
  
  with st.container():
    st.title("Project: Cardionaut")
    st.write("Heart Attack AI predictor")
  
  col1, col2 = st.columns([4,1])

  
  


 
if __name__ == '__main__':
  main()