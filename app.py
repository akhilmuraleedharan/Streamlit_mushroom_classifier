import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def main():
    st.title("Binary Classifcation Web App")
    st.sidebar.title("Binary Classifcation Web App")
    st.markdown("Are your mushrooms edible or poisonous?🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")
    

    st.cache_data(persist=True)
    def load_data():
        data =  pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] =  label.fit_transform(data[col])
        return data


    st.cache_data(persist=True)
    def split(df):
        y =  df.type
        x =  df.drop(columns = ['type'])
        x_train,x_test, y_train,y_test =  train_test_split(x,y, test_size=0.3, random_state=42)
        return x_train,x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion matrix' in metrics_list:
            fig,ax = plt.subplots()
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test,preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
            disp.plot(ax=ax)
            st.pyplot(fig)
        
        if 'ROC Curve' in metrics_list:
            fig,ax = plt.subplots()
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, preds)
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            display.plot(ax=ax)
            st.pyplot(fig)

        if 'Precision Recall Curve' in metrics_list:
            fig,ax = plt.subplots()
            st.subheader("Precision Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, preds)
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot(ax=ax)
            st.pyplot(fig)



    df = load_data()
    x_train,x_test, y_train,y_test = split(df)
    class_names =  ['edible','poisonous']

    st.sidebar.subheader("Choose Classifier")
    classifier =  st.sidebar.selectbox("Classifier List", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))


    if classifier == 'Support Vector Machine (SVM)' :
        st.sidebar.subheader("Model Hyperparameters")
        C =  st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key= 'kernel')
        gamma =  st.sidebar.radio("Gamma (Kernel Coeff)", ("scale","Auto"), key='Gamma' )

        metrics = st.sidebar.multiselect("Performance Metrics",("Confusion matrix","ROC Curve","Precision Recall Curve"))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Support Vector Machine(SVM) Results")
            model =  SVC(C = C, kernel=kernel,gamma= gamma)
            model.fit(x_train,y_train)
            accuracy =  model.score(x_test,y_test)
            preds  =  model.predict(x_test)
            st.write("Accuracy: " ,round(accuracy,2))
            st.write("Precision: ", round(precision_score(y_test,preds,labels=class_names),2))
            st.write("Recall: ", round(precision_score(y_test,preds,labels=class_names),2))
            plot_metrics(metrics)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameter")

        C =  st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum number of iterations: ",100,500,key='max_iter')

        metrics = st.sidebar.multiselect("Performance Metrics",("Confusion matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Logistic Regression Results")
            model =  LogisticRegression(C = C, max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy =  model.score(x_test,y_test)
            preds  =  model.predict(x_test)
            st.write("Accuracy: " ,round(accuracy,2))
            st.write("Precision: ", round(precision_score(y_test,preds,labels=class_names),2))
            st.write("Recall: ", round(precision_score(y_test,preds,labels=class_names),2))
            plot_metrics(metrics)



    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameter")

        n_estimators = st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='n_estimators')
        max_depth = st.sidebar.number_input("Maximum depth of tree",1,20,step=1,key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",(True,False),key='bootstrap')

        metrics = st.sidebar.multiselect("Performance Metrics",("Confusion matrix","ROC Curve","Precision Recall Curve"))
        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Random Forest Results")
            model =  RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy =  model.score(x_test,y_test)
            preds  =  model.predict(x_test)
            st.write("Accuracy: " ,round(accuracy,2))
            st.write("Precision: ", round(precision_score(y_test,preds,labels=class_names),2))
            st.write("Recall: ", round(precision_score(y_test,preds,labels=class_names),2))
            plot_metrics(metrics)


    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)



if __name__ == '__main__':
    main()