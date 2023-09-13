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
  st.subheader('EDA')
  st.write('uploaded dataset:')
  st.write(df.head())

  #1️⃣EDA
  if st.checkbox("shape"): 
    st.write(df.shape)

  if st.checkbox("show values counts") : 
    st.write(df.iloc[:, -1].value_counts())
  
  if st.checkbox('show null sum'):
    st.write(df.isna().sum())
  
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

  # data Visualization
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
  
  #2️⃣ preprocessing
  st.subheader('Preprocessing')
  
  col_to_drop = st.multiselect("what cols you want to drop", df.columns)
  
  df = df.drop(col_to_drop, axis=1)
  if st.checkbox("show columns after drop"): 
    st.write(df.columns.tolist())
  
  # fill nan in numerical
  num_cols = df.select_dtypes(include=['int64', 'float64']).columns
  for col in num_cols:
    if df[col].isnull().any():
      st.markdown(f'**handling null values in {col}**')
      num_na_type = st.selectbox(f'select how to handle null values in {col}', ['none', 'mean', 'median', 'mode'])
      if num_na_type == 'mean':
        df[col].fillna(df[col].mean(), inplace=True)
      elif num_na_type == 'median':
        df[col].fillna(df[col].median(), inplace=True)
      elif num_na_type == 'mode':
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        
  # fill nan in categorical
  cat_cols = df.select_dtypes(include=['object']).columns
  for col in cat_cols:
    if df[col].isnull().any():
      st.markdown(f'**handling null values in {col}**')
      cat_na_type = st.selectbox(f"select how to handle null values in {col}", ['none','most freq', 'add missing class'])
      if cat_na_type == 'most freq':
        most_freq_val = df[col].mode()[0]
        df[col].fillna(most_freq_val, inplace=True)
      elif cat_na_type == 'add missing class':
        df[col].fillna('Missing', inplace=True)
  
  if st.button('show data info after processing'):
    st.write(df.head())
    st.write("columns after fill null values")
    st.write(df.isna().sum())
    
  #3️⃣ ML settings
  st.subheader('ML settings')
  target_var = st.selectbox("select target variable", df.columns)
  if df[target_var].dtype in ['int64', 'float64']:
    task_type = 'regression'
  else:
    task_type = 'classification'
  if st.button('train model'):
    with st.spinner('training models...'):
      setup_data = setup(df, target=target_var, session_id=123)
      models = compare_models()
    st.success("models trained")
    st.write(pull().style.highlight_max(axis=0))
