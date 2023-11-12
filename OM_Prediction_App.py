import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import plotly.express as px
from sklearn.multiclass import OneVsRestClassifier

st.title('Organic Matter Abundance Predictor (OMAP)')
DATA_URL = ('https://raw.githubusercontent.com/FloydNichols97/OMAP/main/ML_Data.csv')
Map_Data = ('https://raw.githubusercontent.com/FloydNichols97/OMAP/main/Spatial_df.csv')

tab1, tab2 = st.tabs(["Model Data & Performance", "Make Prediction"])

with tab1:
    @st.cache_data
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        return data

# Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache_data)")

    if st.checkbox('Show Model Data'):
        st.subheader('Model Data')
        st.write(data)

    data = data.drop(columns=['Sample', 'S% by weight'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    data = data.dropna()

    st.subheader('Total Organic Carbon Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data.TOC, kde=True, edgecolor='k', color='grey')
    plt.tick_params(labelsize=16)
    plt.xlabel("TOC (%)", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    st.pyplot(fig)

# create a list of our conditions
    conditions = [
        (data['TOC'] <= 2.5),
        (data['TOC'] > 2.5) & (data['TOC'] <= 10),
        (data['TOC'] > 10)
        ]
# create a list of the values we want to assign for each condition
    values = ['Low', 'Moderate', 'High']
# create a new column and use np.select to assign values to it using our lists as arguments
    data['Productivity'] = np.select(conditions, values)

#t-SNE
    X = data.drop(columns=['Productivity'])
    y = data['Productivity']
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

# We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components, random_state = 42)
    tsne_result = tsne.fit_transform(X)

# Two dimensions for each of our images
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
    palette = {"Low":"#D81B60",
            "Moderate":"#1E88E5",
            "High":"#FFC107"}

    st.subheader('t-Distributed Stochastic Neighbor Embedding')
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig = px.scatter(tsne_result_df, x='tsne_1', y='tsne_2', color = 'label',  color_discrete_map=palette)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.8, 0.95), loc=2, borderaxespad=0.0)
    st.plotly_chart(fig, use_container_width=True)

# PCA
    X = data.drop(columns=['Productivity'])
    y = data['Productivity']
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)

    st.subheader('Principal Component Analysis')
    PCA_df = pd.DataFrame({'PCA_1': Xt[:, 0], 'PCA_2': Xt[:, 1], 'label': y})
    fig = px.scatter(PCA_df, x='PCA_1', y='PCA_2', color = 'label',  color_discrete_map=palette)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.8, 0.95), loc=2, borderaxespad=0.0)
    st.plotly_chart(fig, use_container_width=True)


# Model Classifier
    st.subheader('Model Performance')
    np.random.seed(300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    # Random Forest Classifier
    RF_model = RandomForestClassifier(n_estimators=40, min_samples_split = 5, bootstrap=True, random_state = 42)
    RF_model.fit(X_train, y_train)
    RF_predictions = RF_model.predict(X_test)
    RF_cv = RepeatedStratifiedKFold(n_splits=20, n_repeats=10, random_state=1)
    RF_scores = cross_val_score(RF_model, X, y, scoring='accuracy', cv=RF_cv, n_jobs=-1, error_score='raise')

# Compute Confusion Matrix
    fig, ax = plt.subplots(1, figsize=(4,4))
    cm = confusion_matrix(y_test, RF_predictions, labels=RF_model.classes_)
    sns.heatmap(cm, annot = True,  cmap=plt.cm.Greens)
# Add labels to the plot
    class_names = RF_model.classes_
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Model')
    st.pyplot(fig)
    #st.write('Accuracy: %.3f (%.3f)' % ((mean(RF_scores), std(RF_scores))))
    st.text(classification_report(y_test, RF_predictions))

with tab2:
    st.subheader("Upload a Data File for Organic Matter Prediction")
    uploaded_file = st.file_uploader("Choose a file")
    def load_data(nrows):
        new_data = pd.read_csv(uploaded_file, nrows=nrows)
        return new_data

# Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
    new_data = load_data(10000)
# Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache_data)")

    if st.checkbox('Show New Data'):
        st.subheader('New Data')
        st.write(new_data)

# create a list of our conditions
    conditions = [
        (new_data['TOC'] <= 2.5),
        (new_data['TOC'] > 2.5) & (new_data['TOC'] <= 10),
        (new_data['TOC'] > 10)
        ]

# create a list of the values we want to assign for each condition
    values = ['Low', 'Moderate', 'High']

# create a new column and use np.select to assign values to it using our lists as arguments
    new_data['Productivity'] = np.select(conditions, values)

    new_data = new_data.drop(columns=['Sample', 'S% by weight'])
    new_data = new_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    new_data = new_data.dropna()

    X_new_data = new_data.drop(columns=['Productivity'])
    y = new_data['Productivity']

# RF
    RF = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, min_samples_split = 5, bootstrap=True, random_state = 1))
    RF.fit(X_train, y_train)
    RF.predict_proba(X_test)
    predict_RF = RF_model.predict(X_new_data)
    probabilities_RF = RF.predict_proba(X_new_data)
    probabilities_RF = pd.DataFrame(probabilities_RF)
    probabilities_RF = probabilities_RF.rename({0: 'High',
                                                1: 'Low',
                                                2: 'Moderate'}, axis = 1)

    fig, ax = plt.subplots(1, figsize=(4,4))
    probabilities_RF.plot(kind='area', stacked=True, color=['#FFC107', '#D81B60', '#1E88E5'],  title = "Random Forest")
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    #st.pyplot(fig)

    st.subheader("Model Predictions and Probabilities")
    st.write(probabilities_RF)