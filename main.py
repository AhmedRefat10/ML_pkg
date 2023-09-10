import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from pycaret.classification import *
from pycaret.regression import *

@st.cache_resource
def load_data(data):
  if data.name.endswith('.csv'):
    df = pd.read_csv(data, nrows=2000)
  else:
    df = pd.read_excel(data, nrows=2000)
  return df
  
# set title and description of the proj
st.title("ML web app using PyCaret and Streamlit")
st.write("upload dataset, Perform EDA, select the target var, and choose models for automatic training")

# upload dataset
data = st.file_uploader("upload a CSV or Excel file", type=['csv', 'xlsx'])

if data:
  df = load_data(data)
  
  st.write('uploaded dataset:')
  st.write(df.head())

  # EDA plots
  st.subheader('EDA')
      
  if st.checkbox("shape"): 
    st.write(df.shape)

  if st.checkbox("show values counts") : 
    st.write(df.iloc[:, -1].value_counts())
              
  if st.checkbox("description"): 
    st.write(df.describe())
            
  if st.checkbox("columns datatypes"):
    dtypes_df = pd.DataFrame({'col': df.dtypes.index, 'type': df.dtypes.values})
    st.write(dtypes_df)
            
  if st.checkbox("columns"): 
    st.write(df.columns.tolist())
    
  if st.checkbox("show specific column"): 
    cols_df = df[st.multiselect('select column', df.columns.tolist())]
    st.dataframe(cols_df)
              
  if st.checkbox("correlation"):
    fig, ax = plt.subplots()
    st.write(sns.heatmap(df.corr(),annot=True))
    st.pyplot(fig)
    plt.close(fig)
    
  # data visualization
  st.subheader('Data visualization')        
  cols_names = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
      
  if cols_names:
    plot_type = st.selectbox("select plot type", ['bar', 'line', 'area', 'hist', 'box'])
    selected_cols = st.multiselect('select cols to plot', cols_names)
      
  if st.button("plot the column"):
    st.success(f"{plot_type} plot for {selected_cols}")
   
  if plot_type == 'bar':
    st.bar_chart(df[selected_cols])
            
  elif plot_type == 'line':
    st.line_chart(df[selected_cols])
            
  elif plot_type == 'area':
    st.area_chart(df[selected_cols])
          
  elif plot_type:
    fig, ax = plt.subplots()
    df[selected_cols].plot(kind=plot_type, ax=ax)
    st.pyplot(fig)
  
  # Ml Settings
  st.subheader("ML settings")
  target_var = st.selectbox("select target var", df.columns)
  supervised_model = st.selectbox('select model', ['none','regression', 'classification'])
  if supervised_model != 'none':
    setup_data = setup(df, target=target_var, session_id=123)
  
    # model Training
    
    if supervised_model == 'classification':
      models = compare_models()
    else:
      models = compare_models(fold=5, round=2, sort="MAE")
    
    # evaluate model
    evaluate_model(models)
    
    # get score
    st.subheader('Model performance')
    test_set = df.copy()
    test_set.drop(target_var, axis=1, inplace=True)
    
    predictions = predict_model(models, data=test_set)
    if supervised_model == 'classification':
      score = accuracy_score(df[target_var], predictions['prediction_label'])
    else:
      score = mean_squared_error(df[target_var], predictions['prediction_label'])
    st.write(f'accuracy : {score:.2f}')
