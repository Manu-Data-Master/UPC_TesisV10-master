import pandas as pd
import pickle # Object serialization
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics
import streamlit as st
from io import StringIO

def load_dataset(csv_data):
    #df = pd.read_csv(csv_data)
    df = csv_data

    # print(f'Top5 datas: \n{df.head()}')
    # print(f'Last5 datas: \n{df.tail()}')
    # print(f'Specific class: \n', df[df['class']=='bridge'])  # Show specific class data.

    #features = df.drop('class', axis=1) # Features, drop the colum 1 of 'class'.
    features = df.drop(['class', 'video', 'frames_per_sec', 'frame_count'], axis=1) # Features, drop the colum 1 of 'class'.
    target_value = df['class']          # target value.

    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.3, random_state=1234)
    # x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.2, random_state=1234)
    # x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.2, random_state=0, stratify=target_value)

    return x_train, x_test, y_train, y_test

def evaluate_model(fit_models, x_test, y_test):
    #print('\nEvaluate model accuracy:')
    st.info('Evaluate model accuracy:', icon='📚')

    # Evaluate and Serialize Model.
    for key_algo, value_pipeline in fit_models.items():
        yhat = value_pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, yhat)*100
        #print(f'Classify algorithm: {key_algo}, Accuracy: {accuracy}%')
        st.success('Classify algorithm: {}, Accuracy: {}%'.format(key_algo, accuracy), icon='🎯')
        st.markdown("<br>", unsafe_allow_html=True)

#if __name__ == '__main__':
def main_function(dataset_csv_file, model_weights):
    #dataset_csv_file = './dataset/coords_dataset_20230315_191810.csv'
    #model_weights = './model_weights/weights_body_language_220230315_191810.pkl'

    x_train = load_dataset(csv_data=dataset_csv_file)[0]
    y_train = load_dataset(csv_data=dataset_csv_file)[2]
    x_test = load_dataset(csv_data=dataset_csv_file)[1]
    y_test = load_dataset(csv_data=dataset_csv_file)[3]
    
    pipelines = {
        'lr' : make_pipeline(StandardScaler(), LogisticRegression()),
        'rc' : make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    #print('key:', pipelines.keys())
    #print('value:', list(pipelines.values())[0]) # 0~3
    st.info('key: {}'.format(pipelines.keys()), icon='📚')
    st.markdown("<br>", unsafe_allow_html=True)
    st.info('value: {}'.format(list(pipelines.values())[0]), icon='📚')
    st.markdown("<br>", unsafe_allow_html=True)

    fit_models = {}
    #print('Model is Training ....')
    st.warning('Training Model...', icon='⚠️')
    st.markdown("<br>", unsafe_allow_html=True)

    for key_algo, value_pipeline in pipelines.items():
        model = value_pipeline.fit(x_train, y_train)
        fit_models[key_algo] = model
    #print('Training done.')
    st.success('Training done!', icon='🎯')
    st.markdown("<br>", unsafe_allow_html=True)

    # Using x_test data input to Ridge Classifier model to predict.
    rc_predict = fit_models['rc'].predict(x_test)
    #print(f'\nPredict 5 datas: {rc_predict[0:5]}')
    st.info('Showing first 5 prediction values: {}'.format(rc_predict[0:5]), icon='📚')
    st.markdown("<br>", unsafe_allow_html=True)

    # Save model weights.
    st.warning('Saving Model...', icon='⚠️')
    st.markdown("<br>", unsafe_allow_html=True)

    with open(model_weights, 'wb') as f:
        # pickle.dump(obj, file, [,protocol=0])
        # 將obj對象序列化存入已經打開的file中。
        pickle.dump(fit_models['rf'], f)
    #print('\nSave model done.')
    st.success('Model saved!', icon='🎯')
    st.markdown("<br>", unsafe_allow_html=True)
    
    evaluate_model(fit_models, x_test, y_test)
