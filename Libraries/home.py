import streamlit as st

def load_home_sidebar(edad, peso, talla, imc, perabdominal, genero):
    #SIDEBAR START    
    st.sidebar.markdown('---')
    st.sidebar.markdown('❤️ __HEALTH INFORMATION__ ', unsafe_allow_html=True)
    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    st.sidebar.markdown('🟥 __Gender:__ {}'.format('Male' if genero == 'M' else 'Female'), unsafe_allow_html=True)
    st.sidebar.markdown('🟧 __Age:__ {:.0f} years old'.format(edad), unsafe_allow_html=True)
    st.sidebar.markdown('🟨 __Weight:__ {:.2f} kg'.format(peso), unsafe_allow_html=True)
    st.sidebar.markdown('🟩 __Height:__ {:.2f} cm'.format(talla), unsafe_allow_html=True)
    st.sidebar.markdown('🟦 __BMI:__ {:.2f} Kg/m<sup>2</sup> (Body Mass Index)'.format(imc), unsafe_allow_html=True)
    st.sidebar.markdown('🟪 __Abdominal circumference:__ {:.2f} cm'.format(perabdominal), unsafe_allow_html=True)
    st.sidebar.markdown('---')

def load_home(edad, peso, talla, imc, perabdominal):
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**POSE_LANDMARKS**<br>Una lista de puntos de referencia de la pose. Cada punto de referencia consta de lo siguiente:<br><ul><li><b>X & Y:</b> coordenadas de referencia normalizadas a [0.0, 1.0] por el ancho y la altura de la imagen, respectivamente.</li><li><b>Z:</b> Representa la profundidad del punto de referencia con la profundidad en el punto medio de las caderas como origen, y cuanto menor sea el valor, más cerca estará el punto de referencia de la cámara. La magnitud de z usa aproximadamente la misma escala que x.</li><li><b>Visibilidad:</b> un valor en [0.0, 1.0] que indica la probabilidad de que el punto de referencia sea visible (presente y no ocluido) en la imagen.</li></ul><br>",
        unsafe_allow_html=True)
    st.markdown("**MODELO DE PUNTOS DE REFERENCIA DE POSE (BlazePose GHUM 3D)**<br>El modelo de puntos de referencia en MediaPipe Pose predice la ubicación de 33 puntos de referencia de pose (consulte la figura a continuación).<br>",
        unsafe_allow_html=True)
    st.image("01. webapp_img/pose_landmarks_model.png", width=600)