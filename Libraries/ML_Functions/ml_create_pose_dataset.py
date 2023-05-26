import pandas as pd
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees
import Libraries.dashboard as dashboard


def main_function(uploaded_png_files, id_exercise):
    n_poses = get_number_poses(id_exercise)
    articulaciones_list = get_articulaciones(id_exercise)
    landmark_list = get_landmark_values(id_exercise)
    
    #mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    path_png_input = './99. testing_resourses/inputs/create_pose_datasets/{}/'.format(id_exercise)
    list_images = []
    for i in uploaded_png_files:
        list_images.append(i.name)
    list_images.sort()

    c = 0
    trainer = 1
    df_puntos_final = pd.DataFrame()
    
    count_temp = 0
    dic_artic_ladmarks = {}
    for art in articulaciones_list:
        list_lnd_to_dict = []
        list_lnd_to_dict.append(landmark_list[count_temp])
        list_lnd_to_dict.append(landmark_list[count_temp + 1])
        list_lnd_to_dict.append(landmark_list[count_temp + 2])
        dic_artic_ladmarks[art] = list_lnd_to_dict
        count_temp += 3

    for img in list_images:
        with mp_pose.Pose(static_image_mode=True) as pose:
            st.markdown("<br >", unsafe_allow_html=True)
            st.markdown("🩻 Processing image: {}{}".format(path_png_input, img), unsafe_allow_html=True)

            image = cv2.imread("{}{}".format(path_png_input, img),1)

            height, width, _ = image.shape
            #transformamos la imagen de entrada a rgb, ya que así las lee open cv
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            #dibujamos los puntos de referencia y sus conexiones
            if results.pose_landmarks is not None:
                resultados = []
                
                for i in range(0, len(results.pose_landmarks.landmark)):
                    resultados.append(results.pose_landmarks.landmark[i].x)
                    resultados.append(results.pose_landmarks.landmark[i].y)
                    resultados.append(results.pose_landmarks.landmark[i].z)
                    resultados.append(results.pose_landmarks.landmark[i].visibility)
                    
                landmarks = results.pose_landmarks.landmark

                dict_body_part_angle = {}
                for artic in dic_artic_ladmarks:
                    body_part_x_y_list = [0, 0, 0]
                    count_bpxyl = 0
                    for landmark_id in dic_artic_ladmarks.get(artic):
                        body_part_x = int(landmarks[int(landmark_id)].x * width)
                        body_part_y = int(landmarks[int(landmark_id)].y * width)
                        body_part_x_y_list[count_bpxyl] = np.array([body_part_x, body_part_y])
                        count_bpxyl += 1
                    right_arm_l1 = np.linalg.norm(body_part_x_y_list[1] - body_part_x_y_list[2])
                    right_arm_l2 = np.linalg.norm(body_part_x_y_list[0] - body_part_x_y_list[2])
                    right_arm_l3 = np.linalg.norm(body_part_x_y_list[0] - body_part_x_y_list[1])
                    body_part_angle = degrees(acos((right_arm_l1**2 + right_arm_l3**2 - right_arm_l2**2) / (2*right_arm_l1*right_arm_l3)))
                    dict_body_part_angle[artic] = body_part_angle
                
                df_puntos = pd.DataFrame(np.reshape(resultados, (132, 1)).T)
                df_puntos['pose'] = c + 1

                for body_part_angle in dict_body_part_angle:
                    print(body_part_angle)
                    df_puntos[body_part_angle] = dict_body_part_angle.get(body_part_angle)
                
                df_puntos_final = pd.concat([df_puntos_final, df_puntos])

            c += 1
            if c % n_poses == 0:
                df_puntos_final_columns = ["nose_x","nose_y","nose_z","nose_visibility",
                                           "left_eye_inner_x","left_eye_inner_y","left_eye_inner_z","left_eye_inner_visibility",
                                           "left_eye_x","left_eye_y","left_eye_z","left_eye_visibility",
                                           "left_eye_outer_x","left_eye_outer_y","left_eye_outer_z","left_eye_outer_visibility",
                                           "right_eye_inner_x","right_eye_inner_y","right_eye_inner_z","right_eye_inner_visibility",
                                           "right_eye_x","right_eye_y","right_eye_z","right_eye_visibility",
                                           "right_eye_outer_x","right_eye_outer_y","right_eye_outer_z","right_eye_outer_visibility",
                                           "left_ear_x","left_ear_y","left_ear_z","left_ear_visibility",
                                           "right_ear_x","right_ear_y","right_ear_z","right_ear_visibility",
                                           "mouth_left_x","mouth_left_y","mouth_left_z","mouth_left_visibility",
                                           "mouth_right_x","mouth_right_y","mouth_right_z","mouth_right_visibility",
                                           "left_shoulder_x","left_shoulder_y","left_shoulder_z","left_shoulder_visibility",
                                           "right_shoulder_x","right_shoulder_y","right_shoulder_z","right_shoulder_visibility",
                                           "left_elbow_x","left_elbow_y","left_elbow_z","left_elbow_visibility",
                                           "right_elbow_x","right_elbow_y","right_elbow_z","right_elbow_visibility",
                                           "left_wrist_x","left_wrist_y","left_wrist_z","left_wrist_visibility",
                                           "right_wrist_x","right_wrist_y","right_wrist_z","right_wrist_visibility",
                                           "left_pinky_x","left_pinky_y","left_pinky_z","left_pinky_visibility",
                                           "right_pinky_x","right_pinky_y","right_pinky_z","right_pinky_visibility",
                                           "left_index_x","left_index_y","left_index_z","left_index_visibility",
                                           "right_index_x","right_index_y","right_index_z","right_index_visibility",
                                           "left_thumb_x","left_thumb_y","left_thumb_z","left_thumb_visibility",
                                           "right_thumb_x","right_thumb_y","right_thumb_z","right_thumb_visibility",
                                           "left_hip_x","left_hip_y","left_hip_z","left_hip_visibility",
                                           "right_hip_x","right_hip_y","right_hip_z","right_hip_visibility",
                                           "left_knee_x","left_knee_y","left_knee_z","left_knee_visibility",
                                           "right_knee_x","right_knee_y","right_knee_z","right_knee_visibility",
                                           "left_ankle_x","left_ankle_y","left_ankle_z","left_ankle_visibility",
                                           "right_ankle_x","right_ankle_y","right_ankle_z","right_ankle_visibility",
                                           "left_heel_x","left_heel_y","left_heel_z","left_heel_visibility",
                                           "right_heel_x","right_heel_y","right_heel_z","right_heel_visibility",
                                           "left_foot_index_x","left_foot_index_y","left_foot_index_z","left_foot_index_visibility",
                                           "right_foot_index_x","right_foot_index_y","right_foot_index_z","right_foot_index_visibility", 
                                           "pose"]
                
                df_puntos_final_columns.extend(articulaciones_list)
                df_puntos_final.columns = df_puntos_final_columns
                
                first_column = df_puntos_final.pop('pose')
                df_puntos_final.insert(0, 'pose', first_column)
                
                st.markdown("<br >", unsafe_allow_html=True)
                path_png_output = './99. testing_resourses/outputs/create_pose_datasets/{}/'.format(id_exercise)
                files_csv = "{}_puntos_trainer{}.csv".format(id_exercise, trainer)

                st.markdown("📅 Generate trainer CSV file: {}{}".format(path_png_output, files_csv), unsafe_allow_html=True)
                st.dataframe(df_puntos_final)                
                df_puntos_final.to_csv("{}{}".format(path_png_output, files_csv), index=False)
                df_puntos_final = pd.DataFrame()
                trainer += 1
                c = 0
                st.markdown("------", unsafe_allow_html=True)
    

def list_exercise():
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    list_exercises = df_exercise['id_exercise'].unique().tolist()
    return list_exercises

def get_number_poses(id_exercise):
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    df_exercise = df_exercise.loc[df_exercise['id_exercise'] == id_exercise]

    return df_exercise['n_poses'].loc[df_exercise.index[0]]

def get_articulaciones(id_exercise):
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    df_exercise = df_exercise.loc[df_exercise['id_exercise'] == id_exercise]

    articulaciones =  df_exercise['articulaciones'].loc[df_exercise.index[0]]
    articulaciones_broken = dashboard.get_articulaciones_list(articulaciones)
    
    return articulaciones_broken


def get_landmark_values(id_exercise):
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    df_exercise = df_exercise.loc[df_exercise['id_exercise'] == id_exercise]

    landmark_values = df_exercise['landmark_values'].loc[df_exercise.index[0]]
    landmark_values_broken = dashboard.get_articulaciones_list(landmark_values)

    return landmark_values_broken
