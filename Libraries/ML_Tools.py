import streamlit as st
import pandas as pd
import time

import Libraries.ML_Functions.ml_create_pose_dataset as ml_crea_pos_dt
import Libraries.ML_Functions.ml_model_train as ml_model_train
import Libraries.ML_Functions.ml_model_test as ml_model_test

def load_ml_tools():
    tab_create_pose_dataset, tab_ml_model_train, tab_ml_model_test = st.tabs(["1️⃣💾 CREATE POSE DATASET", "2️⃣🤖 ML MODEL TRAINING", "3️⃣🦾 ML MODEL TEST"])
    with tab_create_pose_dataset:
        st.markdown("**💾 CREATE POSE DATASET**", unsafe_allow_html=True)
        st.markdown("Crea dataset de pose en base a :")
        st.markdown("<br>", unsafe_allow_html=True)
        id_exercise = st.selectbox("Choose exercise", ml_crea_pos_dt.list_exercise())
        uploaded_png_files = st.file_uploader("Choose a PNG file for Trainer 1", type= ['png'], accept_multiple_files=True )
        ml_crea_pos_dt.main_function(uploaded_png_files, id_exercise)
        #st.text(str(uploaded_png_files))
    
    
    
    
    with tab_ml_model_train:
        st.markdown("**🤖 ML MODEL TRAINING**", unsafe_allow_html=True)
        st.markdown("Genera modelo entrenado de ML :", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("1️⃣ UPLOAD CSV COORDS DATASET FILE :", unsafe_allow_html=True)
        uploaded_csv_file = st.file_uploader("Choose a CSV file", type= ['csv'] )
        if uploaded_csv_file is not None:
            dataframe = pd.read_csv(uploaded_csv_file, sep=',')
            st.dataframe(dataframe)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("2️⃣ MODEL PKL FILE TO BE GENERATED:", unsafe_allow_html=True)
            pkl_time_stamp = time.strftime("%Y%m%d_%H%M%S")
            #path_pkl = './model_weights/'
            path_pkl = './99. testing_resourses/outputs/'
            file_pkl = 'weights_body_language_{}.pkl'.format(pkl_time_stamp)
            path_file_pkl = path_pkl + file_pkl

            st.text_area(label='', value = '{}'.format(path_file_pkl))
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("3️⃣TRAINING MODEL :", unsafe_allow_html=True)
            ml_model_train.main_function(dataframe, path_file_pkl)

    with tab_ml_model_test:
        st.markdown("**🦾 ML MODEL TEST**", unsafe_allow_html=True)
        st.markdown("Testea modelo de ML :", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("1️⃣ UPLOAD PNG IMAGE FILE :", unsafe_allow_html=True)
        uploaded_png_file = st.file_uploader("Choose a PNG file", type= ['png'] )        
        path_png = './99. testing_resourses/inputs/'
        
        if uploaded_png_file is not None:
            st.image(uploaded_png_file, caption='Model Test Image', width=400)
            uploaded_png_file_name = uploaded_png_file.name
            path_file_png = path_png + uploaded_png_file_name
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("2️⃣ SELECT MODEL PKL FILE TO TEST:", unsafe_allow_html=True)
            list_pkl = ml_model_test.get_files_by_dir('./99. testing_resourses/outputs/', 'pkl')
            file_pkl = st.selectbox('Select model to test', list_pkl, index=0)     
            
            btn_test_model = st.button("Test Model")
            
            if (btn_test_model):
                #path_pkl = './model_weights/'
                path_pkl = './99. testing_resourses/outputs/'
                path_file_pkl = path_pkl + file_pkl

                ml_model_test.main_function(path_file_png, path_file_pkl)