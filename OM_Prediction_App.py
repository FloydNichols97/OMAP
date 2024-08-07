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
#DATA_URL = ('https://raw.githubusercontent.com/FloydNichols97/OMAP/main/ML_Data.csv')
DATA_URL = ('https://raw.githubusercontent.com/FloydNichols97/OMAP/main/ML_Data_2.csv')
Map_Data = ('https://raw.githubusercontent.com/FloydNichols97/OMAP/main/Spatial_df2.csv')

tab1, tab2, tab3, tab4 = st.tabs(["Information", "Model Data & Performance", "Make Prediction", "Geographic Distribution"])

with tab1:
    st.subheader('Synopsis')
    st.markdown('''This page allows on-demand prediction of estimated organic matter abundance from elemental abundance data. It is intended as a tool to provide quick estimates of organic matter abundance which requires more time and resources than acquiring elemental abundance data. This tool will be especially relevant for sample selection on Mars. Currently, estimates are calculated from a series of Mars analog hypersaline lakes according to an algorithm employed by Nichols et al. (2024?). This papers is the primary references for the OMAP and provides description of the methods.''')
    st.subheader('Instructions')
    st.markdown('''To improve accessibility of the model and provide rapid prediction of OC abundance from 
XRF-derived elemental abundances, we have developed an open-source graphical user interface: 
Organic Matter Abundance Predictor (OMAP; Nichols, 2024). This application is an interactive 
data visualization tool and predictor for the constructed model. Due to the random forest algorithm 
having the best performance for our testing and validation, we use it as the primary algorithm for 
the application. The application consists of three main components including model data and 
performance visualization, organic carbon probability predictor, and geographic distribution of 
samples that the model is based on. Additionally, this application serves as an open-source 
database for others to add OC and XRF-derived elemental abundances from other sedimentary 
systems to improve and expand upon the model.
Model Set Up and View Model Data and Performance
This section was designed to provide transparency and give the user the ability to view the 
data used to run the model in addition to tuning it to their own needs. Considering that not all XRF 
instruments can analyze the same elements as presented in this paper, we allow the user to select 
the elements of interest(s) in Select Elements. The default elements selected are all available 
elements that the model is capable of running; however, if the user can only analyze fewer 
elements, then they can clear the default elements using the right ‘x’ button and manually select 
the elements of interest. 
Once the desired elements are selected, the user may define the boundary conditions for 
organic matter abundance classification. The default boundary conditions are 2.5 for low and 10.0 
for high as described in the main text of this paper. Once the boundary conditions are set, the 
application will automatically produce two interactive figures for dimensionality reduction of the 
data and model metrics including accuracy, recall, precision, and F1 score.
Additionally, this model was designed to look at patterns between elemental abundances 
and ratios, as such the model is capable of making predictions for variety of instrumentation that 
can resolve elemental abundance concentrations and ratios such as Gamma-ray spectroscopy as 
described in the main text of this paper. Similarly, the original model takes into account different 
resolutions of sedimentary sampling and as such can be compared at 0.5 cm scale to broader scales 
of 3 cm resolutions.
Load and Analyze
Once the model is set for the desired elements and boundary conditions as described above, 
the user can now make a prediction of their own elemental abundance dataset. To do so, click the 
‘Browse files’ button and select a ‘.csv’ file of the elemental abundance data. It is important that 
the csv file is set up properly for the model to make a prediction. Example set up below. 
Additionally, if the following error is received: ‘KeyError: This app has encountered an error. The 
original error message is redacted to prevent data leaks. Full error details have been recorded in 
the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app)’ after 
uploading the file then the elements selected on the model set up page do not match the csv file. 
To solve the error, return to the model set up page and make sure that only the elements present in 
your csv file are selected.
''')
    st.subheader('Contact')
    st.write('floydnichols@vt.edu')

with tab2:
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

    data = data.drop(columns=['Sample'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    data = data.dropna()
    elements = st.multiselect("Select elements (default is all):", data.columns, default = ['Mg', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se',
                    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ag', 'Cd', 'Sn', 'Sb', 'W', 'Hg',
                    'Pb', 'Bi', 'Th', 'U', 'LE', 'Al', 'Si', 'P', 'S', 'K', 'Ca'])

    st.header('Total Organic Carbon Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data.TOC, kde=True, edgecolor='k', color='grey')
    plt.tick_params(labelsize=16)
    plt.xlabel("TOC (%)", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    st.pyplot(fig)

    st.header('Dimensionality Reduction')
    st.markdown('''Select boundary conditions for OM classes. For a binary classification only input a number for Low OM upper cutoff. For a 3-class model, input a lower boundary number for High OM.''')
    low = st.number_input('Insert Upper Cutoff for Low OM', min_value=0.5)
    st.write("Original Model Value = 2.5")
    high = st.number_input('Insert Lower Cutoff for High OM')
    st.write("Original Model Value = 10.0")
    st.subheader('t-Distributed Stochastic Neighbor Embedding')

    # create a list of our conditions
    conditions = [
        (data['TOC'] <= low),
        (data['TOC'] > low) & (data['TOC'] <= high),
        (data['TOC'] > high)
        ]
    # create a list of the values we want to assign for each condition
    values = ['Low', 'Moderate', 'High']
    # create a new column and use np.select to assign values to it using our lists as arguments
    data['Productivity'] = np.select(conditions, values)

    # t-SNE
    X = data[elements] # Make prediction based on selected elements
    #X = data.drop(columns=['Productivity'])
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

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig = px.scatter(tsne_result_df, x='tsne_1', y='tsne_2', color = 'label',  color_discrete_map=palette)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='Black')))
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.8, 0.95), loc=2, borderaxespad=0.0)
    st.plotly_chart(fig, use_container_width=True)

    # PCA
    X = data[elements] # Make prediction based on selected elements
    #X = data.drop(columns=['Productivity'])
    y = data['Productivity']
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)

    st.subheader('Principal Component Analysis')
    PCA_df = pd.DataFrame({'PCA_1': Xt[:, 0], 'PCA_2': Xt[:, 1], 'label': y})
    fig = px.scatter(PCA_df, x='PCA_1', y='PCA_2', color = 'label',  color_discrete_map=palette)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='Black')))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.8, 0.95), loc=2, borderaxespad=0.0)
    st.plotly_chart(fig, use_container_width=True)

    # Model Classifier
    st.header('Model Performance')
    data = data.drop(columns=['TOC'])
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

with tab3:
    st.subheader("Upload a Data File for Organic Matter Prediction")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.error('Need to upload a file')
    else:
        def load_data(nrows):
         new_data = pd.read_csv(uploaded_file, nrows=nrows)
         return new_data

        data_load_state = st.text('Loading data...')
        new_data = load_data(10000)
        data_load_state.text("Done!")

        if st.checkbox('Show New Data'):
            st.subheader('New Data')
            st.write(new_data)

        new_data = new_data.drop(columns=['Sample'])
        new_data = new_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        new_data = new_data.dropna()

        X_new_data = new_data[elements] # Make prediction based on selected elements

        # RF
        RF = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, min_samples_split=5, bootstrap=True, random_state=1))
        RF.fit(X_train, y_train)
        RF.predict_proba(X_test)
        predict_RF = RF_model.predict(X_new_data)
        probabilities_RF = RF.predict_proba(X_new_data)
        probabilities_RF = pd.DataFrame(probabilities_RF)
        probabilities_RF = probabilities_RF.rename({0: 'High',
                                                    1: 'Low',
                                                    2: 'Moderate'}, axis=1)

        fig, ax = plt.subplots(1, figsize=(4, 4))
        probabilities_RF.plot(kind='area', stacked=True, color=['#FFC107', '#D81B60', '#1E88E5'], title="Random Forest")
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        # st.pyplot(fig)

        st.subheader("Model Predictions and Probabilities")
        st.write(probabilities_RF)


with tab4:
    Map_Data_df = {'lat': [50.6, 50.59, 50.58, 51.07, 51.32, 61.13],
                   'lon': [-121.35, -121.34, -121.34, -121.58, -121.63, -45.34]}
    Map_Data_df = pd.DataFrame(data= Map_Data_df)
    st.markdown('''This tab displays the current geographic distribution of lakes represented in the model. This model was originally developed using Mars-analog hypersaline lakes in British Columbia; however, the authors encourage more data from other hypersaline systems to represent a greater diversity of regions and improve the model. Please see the contact on the 'Information' tab if you are interested in contributing data to the model. ''')
    st.map(Map_Data_df, latitude = "lat", longitude = "lon", color = '#0000FF', size = 1000)
