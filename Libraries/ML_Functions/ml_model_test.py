import pickle
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
import streamlit as st

def img_model_test(image, model):

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )    

        # Extract Pose landmarks
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        # Concate rows
        row = pose_row

        # Make Detections
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        #print(f'class: {body_language_class}, prob: {body_language_prob}')

        return body_language_class, body_language_prob
    
def get_files_by_dir(dir, ext):
   c_whole_training = []
   for file in os.listdir(dir):
      if file.endswith(".{}".format(ext)):
         c_whole_training.append(file)
   return c_whole_training

#if __name__ == '__main__':
def main_function(image_path, model_weights):
    #img = cv2.imread('./resource/imgs/test5.png')
    #model_weights = './model_weights/weights_body_language.pkl'

    img = cv2.imread(image_path)
    
    # Load Model.
    with open(model_weights, 'rb') as f:
        model = pickle.load(f)

    # Input image to test model predict. 
    predict_class = img_model_test(img, model)[0]
    class_prob = img_model_test(img, model)[1]

    #print(f'\nclass: {predict_class}, acc: {class_probbile}\n')
    st.success('class: {}, acc: {}'.format(predict_class, class_prob), icon='🎯')
