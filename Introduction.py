import streamlit as st

st.write("""
            ## **Machine Learning App**

            The ***missing values*** are imputed with the ***mean*** for each column. The ***data types*** are automatically assigned by default.

            **Polynomial Regression:** 

            Computes upto 6th degree polynomial & selects the nth degree model with highest test score.


            **Decision Tree (RandomizedSearchCV):**

            ***simple:*** 
            parameters={"regressor__splitter":["best"],
                "regressor__max_depth" : [int(x) for x in np.linspace(2,15,10)],
              "regressor__min_samples_leaf":[int(x) for x in np.linspace(1,5,3)],
              "regressor__max_features":["auto"] }
              
            ***exhaustive:***
            parameters={"regressor__splitter":["best","random"],
                "regressor__max_depth" : [int(x) for x in np.linspace(2,15,12)],
              "regressor__min_samples_leaf":[int(x) for x in np.linspace(1,5,3)],
              "regressor__max_features":["auto","log2","sqrt",None] }


            **Classification Algorithms** 

            - Classification Report for both train and test data is calculated

            - Confusion Matrix is plotted with % of correct & incorrect predictions

            ***Once model has been trained, upload test data and download the predicted values***

            For regression models, predict was used to get output for download.

            For classification models, predict & predict_proba are used to get two outputs for download.


            *RandomizedSearchCV used for other algorithms similarly*
            
            ## **Seaborn App**
            The following plots are included in the app with widgets based on seaborn documentation: 
            - Relational Plot: https://seaborn.pydata.org/generated/seaborn.relplot.html
            - Distribution Plot: https://seaborn.pydata.org/generated/seaborn.displot.html
            - Categorical Plot: https://seaborn.pydata.org/generated/seaborn.catplot.html
            - Pair Plot: https://seaborn.pydata.org/generated/seaborn.pairplot.html
            - Joint Plot: https://seaborn.pydata.org/generated/seaborn.jointplot.html
            - Heat Map: https://seaborn.pydata.org/generated/seaborn.heatmap.html

            Once the plot is generated, it can be downloaded at dpi=120. So for a 8x8 image, the output resolution will be 960x960 pixels.            
         """)
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

st.sidebar.write('### **About**')
st.sidebar.info(
 """
            Created by:
            [Parthasarathy Ramamoorthy](https://www.linkedin.com/in/parthasarathyr97/) (Analytics Specialist @ Premium Peanut LLC)
        """)
