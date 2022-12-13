import streamlit as st
import joblib
import os
import numpy as np

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def run_ml_app():
    st.subheader("머신러닝 예측")

    dec_temp1 = '''
    
    <br/>

    ---
    
    '''

    st.markdown(dec_temp1, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('값을 입력하세요!! (0~10)')
        sepal_length = st.number_input('Sepal Length', min_value=0.00, max_value=10.00)
        sepal_width = st.number_input('Sepal Width', min_value=0.00, max_value=10.00)
        petal_length = st.number_input('Pepal Length', min_value=0.00, max_value=10.00)
        petal_width = st.number_input('Pepal Width', min_value=0.00, max_value=10.00)
        sample = [sepal_length, sepal_width, petal_length, petal_width]


    with col2:
        st.subheader('예측 결과!!')
        single_sample = np.array(sample).reshape(1,-1)


        # 모델
        model = load_model('models\logistic_regression_model_iris_221208.pkl')

        prediction = model.predict(single_sample)
        pred_prob = model.predict_proba(single_sample)



        if prediction == 0:
            with st.expander('확률확인'):
                st.write('Setosal 확률 : ',np.round(pred_prob[0][0]*100,2), "%") 
                st.write('Vesicolor 확률 : ',np.round(pred_prob[0][1]*100,2), "%")
                st.write('Virginica 확률 : ',np.round(pred_prob[0][2]*100,2), "%")
            st.success('Setosa!')
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/220px-Irissetosa1.jpg')
        elif prediction == 1:
            with st.expander('확률확인'):
                st.write('Setosal 확률 : ',np.round(pred_prob[0][0]*100,2), "%") 
                st.write('Vesicolor 확률 : ',np.round(pred_prob[0][1]*100,2), "%")
                st.write('Virginica 확률 : ',np.round(pred_prob[0][2]*100,2), "%")
            st.success('Versicolor!')
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/220px-Blue_Flag%2C_Ottawa.jpg')
        elif prediction == 2:
            with st.expander('확률확인'):
                st.write('Setosal 확률 : ',np.round(pred_prob[0][0]*100,2), "%") 
                st.write('Vesicolor 확률 : ',np.round(pred_prob[0][1]*100,2), "%")
                st.write('Virginica 확률 : ',np.round(pred_prob[0][2]*100,2), "%")
            st.success('Virginica!')
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/220px-Iris_virginica_2.jpg')
        else:
            pass