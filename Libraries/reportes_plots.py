import plotly.express as px
import streamlit as st
from sklearn import linear_model

def summary_time_plot(df_whole_training, y_column, y_label, barmode, n_palette):

    colorList_1 = [px.colors.qualitative.Alphabet[6], px.colors.qualitative.Alphabet[11], px.colors.qualitative.Plotly[2], px.colors.qualitative.Plotly[7], px.colors.qualitative.G10[5]]
    colorList_2 = ["#7CEA9C", '#50B2C0', "rgb(114, 78, 145)", "hsv(348, 66%, 90%)", "hsl(45, 93%, 58%)"]

    color_pallete_list = [colorList_1, colorList_2]

    total_workout_time = df_whole_training[y_column].sum()
    fig = px.bar(df_whole_training, x='Date_Start', y=y_column,
                 color='id_exercise', 
                 text_auto=True,
                 color_discrete_sequence = color_pallete_list[n_palette],
                 labels={'id_exercise':'Workout routine', 'Date_Start': 'Date', y_column: y_label}, height=400)
    fig.update_layout(
        title = 'WORKOUT TRAINING TIME by EXERCISE ROUTINE - {:.2f} {}'.format(total_workout_time, y_label),
        xaxis_tickformat = '%d %B (%a)<br>%Y',
        plot_bgcolor="#444",
        paper_bgcolor="#444",
        barmode = barmode,
        font_size=16,
    )
    return fig

def scatter_plot(df_whole_training):
    fig = px.scatter(df_whole_training, x="DateTime_Start", y="Prob", color="id_exercise", size='Kcal_factor'
                     #size='petal_length', hover_data=['petal_width']
                     )
    
    fig.update_layout(
        title = 'WORKOUT TRAINING TIME by EXERCISE ROUTINE - ',
        xaxis_tickformat = '%H~%M<br>%d %B (%a)<br>%Y',
        plot_bgcolor="#444",
        paper_bgcolor="#444",
        font_size=16,
    )

    return fig

def regression_plot(df_whole_training, y_column):
        df_whole_training = df_whole_training[[ 'Date_Start', y_column]]
        df_whole_training = df_whole_training.groupby(['Date_Start'], as_index=False)[y_column].sum()
        df_whole_training = df_whole_training.round(2)
        st.dataframe(df_whole_training)
        fig = px.line(df_whole_training, x='Date_Start', y=y_column, text=y_column, markers=True, color_discrete_sequence =['cyan'])
       
        fig.update_layout(
            title = 'WORKOUT TRAINING TIME by EXERCISE ROUTINE - ',
            xaxis_tickformat = '%d %B (%a)<br>%Y',
            plot_bgcolor="#444",
            paper_bgcolor="#444",
            font_size=16,

        )
        return fig