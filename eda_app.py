import streamlit as st
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_data(path):
    # CSV파일
    df = pd.read_csv(path)
    return df

def run_eda_app():
    
    dec_temp1 = '''
    
    ---

    <br/>
    
    '''

    dec_temp2 = '''
    
    <br/>
    
    '''

    # 데이터 경로
    PATH = 'data/iris.csv'
    iris_df = load_data(PATH)

    # side menu
    sidemenu = st.sidebar.selectbox("Side Menu", ['자료분석','시각화'])

    if sidemenu == '자료분석':
        st.subheader('자료 분석')
        st.markdown(dec_temp1, unsafe_allow_html=True)

        with st.expander('데이터 기본정보'):
            st.markdown(utils.attrib_info)

        st.markdown(dec_temp2, unsafe_allow_html=True)

        with st.expander('전체 데이터'):
            st.dataframe(iris_df, width=1000, height=300)
        st.markdown(dec_temp2, unsafe_allow_html=True)

        with st.expander('데이터타입'):
            df2 = pd.DataFrame(iris_df.dtypes).transpose()
            df2.index = ['구분']
            st.dataframe(df2, width=1000, height=200)
        st.markdown(dec_temp2, unsafe_allow_html=True)

        with st.expander('기술 통계량'):
            st.dataframe(pd.DataFrame(iris_df.describe().transpose()), width=1000, height=200)
        st.markdown(dec_temp2, unsafe_allow_html=True)
    
        with st.expander('타겟분포'):
            st.dataframe(iris_df['species'].value_counts())
        st.markdown(dec_temp2, unsafe_allow_html=True)

    elif sidemenu == '시각화':
        st.subheader('시각화')
        st.markdown(dec_temp1, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(['산점도', '히스토그램', '박스플롯', '막대그래프'])
        with tab1:
            st.subheader('IRIS 종별 산점도')
            st.write('IRIS 종을 선택하세요')
            val_species = st.selectbox('', ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), key='1')



            result = iris_df[iris_df['species'] == val_species]

            fig = px.scatter(  result, title = val_species,
                                x='sepal_width', 
                                y='sepal_length', 
                                size='petal_width', 
                                hover_data=['petal_width'])
            st.plotly_chart(fig)


        with tab2:
            st.subheader('데이터별 히스토그램')
            st.write('Sepal & Petal 값을 선택하세요')
            val_hist = st.selectbox('', ('sepal_length', 'sepal_width', 'petal_length','petal_width'), key='2')

            fig, ax = plt.subplots()
            plt.title(val_hist)
            ax.hist(iris_df[val_hist], color='gray')
            st.pyplot(fig)

        with tab3:
            st.subheader('데이터별 박스플롯')
            st.write('Sepal & Petal 값을 선택하세요')

            val_box = st.selectbox('', ('sepal_length', 'sepal_width', 'petal_length','petal_width'), key='3')

            fig, ax = plt.subplots()
            plt.title('Box Plot')
            sns.boxplot(iris_df, x = 'species', y = val_box, ax = ax)
            st.pyplot(fig)

        with tab4:
            st.subheader('데이터별 막대그래프')
            st.write('Sepal & Petal 값을 선택하세요')
            val_bar = st.selectbox('', ('sepal_length', 'sepal_width', 'petal_length','petal_width'), key='4')

            fig, ax = plt.subplots()
            plt.title('Bar Plot')
            sns.barplot(x='species', y=val_bar, data=iris_df)
            st.pyplot(fig)         

    else:
        pass