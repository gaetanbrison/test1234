import streamlit as st
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression




st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)



MODELS = {
    "Logistic Regression":LogisticRegression,
    "KNN":KNeighborsClassifier,
    "Decision Tree":DecisionTreeClassifier,
}

model_mode=st.sidebar.selectbox("Select a model of your choice",['Logistic Regression','KNN','Decision Tree'])

df = pd.read_csv("churn-bigml-20.csv")
st.dataframe(df)


df.State=pd.Categorical(df['State']).codes
df['International plan']=pd.Categorical(df['International plan']).codes
df['Voice mail plan']=pd.Categorical(df['Voice mail plan']).codes
df['Churn']=pd.Categorical(df['Churn']).codes



X = df.drop(["Churn"], axis=1)
y = df.Churn

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


if model_mode == 'Logistic Regression':
    log = LogisticRegression()
    model = log.fit(X_train,y_train)
    prediction = log.predict(X_test)
    st.write("Accuracy Logistic Regression:",metrics.accuracy_score(y_test,prediction))

elif model_mode == 'KNN':
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    prediction = knn.predict(X_test)
    st.write("Accuracy KNN:",metrics.accuracy_score(y_test,prediction))


elif model_mode == 'Decision Tree':
    tree=DecisionTreeClassifier()
    tree.fit(X_train,y_train)
    prediction = tree.predict(X_test)
    st.write("Accuracy Tree:",metrics.accuracy_score(y_test,prediction))