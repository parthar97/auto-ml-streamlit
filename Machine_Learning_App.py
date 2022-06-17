import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pyarrow import csv
import numpy as np

import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import scale 
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from warnings import simplefilter
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

st.write("## **Machine Learning App** ")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV/XLSX file", type=["csv","xlsx"])
@st.experimental_memo
def read_file(uploaded_file):
    try:
        dat=csv.read_csv(uploaded_file)
        data2=dat.to_pandas()  
    except:
        data2=pd.read_excel(uploaded_file)

    quant= [col for col in data2.columns if data2.dtypes[col]!='object']
    qual= [col for col in data2.columns if data2.dtypes[col]=='object']
    quall= [col for col in data2.columns if data2.dtypes[col]=='object' and data2[col].isna().any()!=True]        
    quali=list()
    for i in range(len(qual)):
        if len(set(data2[qual[i]]))<=100:
            quali.append(qual[i])
    qual=quali
    quant1=[0*1 for i in range(len(quant)+1)]
    for i in range(len(quant1)):
            if i==0:
                quant1[i]=None
            else:
                quant1[i]=quant[i-1]
    qual1=[0*1 for i in range(len(qual)+1)]
    for i in range(len(qual1)):
            if i==0:
                qual1[i]=None
            else:
                qual1[i]=qual[i-1]

    #### clean data ####
    for col in data2.columns:
      if pd.api.types.is_numeric_dtype(data2[col]):
               data2[col]=data2[col].fillna((data2[col].mean()))
      else:
               data2[col]=data2[col].fillna('N/A')

    return data2,quant,qual,quant1,qual1,quall

try:
    data2,quant,qual,quant1,qual1,quall=read_file(uploaded_file)

    reg_cla=st.selectbox('Choose Dimensionality Reduction/Regression/Classification',[None,'Dimensionality Reduction','Regression','Classification'])

    if reg_cla=='Dimensionality Reduction':
        container = st.container()
        all = st.checkbox("Select all")
        all_col=[col for col in data2.columns]
        not_quant_col=list(set(all_col)-set(quant))
        if all:
            xx = container.multiselect("Select X:",all_col,all_col)
        else:
            xx = container.multiselect("Select X:",all_col)
        yy=st.multiselect('Select Y',[y for y in quant if y not in xx])
        test_size=st.sidebar.slider('Test Size',0.0,0.5,0.2)
        reg_type=[None,'Principal Component Analysis']
        regtype=st.selectbox('Choose Algorithm',reg_type)
        for col in not_quant_col:
            le = preprocessing.LabelEncoder()
            le.fit(data2[col])
            data2.loc[:, col] = le.transform(data2[col])    

    if reg_cla=='Regression':
        container = st.container()
        all = st.checkbox("Select all")
        all_col=[col for col in data2.columns]
        not_quant_col=list(set(all_col)-set(quant))
        if all:
            xx = container.multiselect("Select X:",all_col,all_col)
        else:
            xx = container.multiselect("Select X:",all_col)
        yy=st.multiselect('Select Y',[y for y in quant if y not in xx])
        test_size=st.sidebar.slider('Test Size',0.0,0.5,0.2)
        reg_type=[None,'Polynomial Regression','Decision Tree',
          'Random Forest','KNN','Lasso','PLS','Elastic Net']
        regtype=st.selectbox('Choose Algorithm',reg_type)
        for col in not_quant_col:
            le = preprocessing.LabelEncoder()
            le.fit(data2[col])
            data2.loc[:, col] = le.transform(data2[col])    

    if reg_cla=='Classification':
        container = st.container()
        all = st.checkbox("Select all")

        if all:
            xx = container.multiselect("Select X:",quant,quant)
        else:
            xx = container.multiselect("Select X:",quant)
        yy=st.selectbox('Select Y',qual1)
        test_size=st.sidebar.slider('Test Size',0.0,0.5,0.2)
        reg_type=[None,'Logistic Regression','KNN Classifier','Gaussian NB Classifier',
              'Decision Tree Classifier','Random Forest Classifier','Support Vector Classifier','Extra Trees Classifier']
        regtype=st.selectbox('Choose Algorithm',reg_type)

    @st.experimental_memo
    def read_col(data2,quant,xx,yy,test_size,reg_cla):    
        random.seed(123)
        X=pd.DataFrame(data2[xx])
        y=pd.DataFrame(data2[yy])
        if reg_cla=='Classification':
            try:
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=0)
            except:
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=data2[qual[1]],random_state=0)                
        if reg_cla=='Regression':
            try:
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=data2[qual[0]],random_state=0)    
            except:
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=data2[qual[1]],random_state=0)                    
        if reg_cla=='Dimensionality Reduction':
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=0)                        
        return X,y,X_train,X_test,y_train,y_test

    X,y,X_train,X_test,y_train,y_test =read_col(data2,quant,xx,yy,test_size,reg_cla)
    st.sidebar.write('Train Data Length : ', len(X_train))
    st.sidebar.write('Test Data Length : ', len(X_test))

    st.write("###### **Dataset**")
    st.write(data2.head())

    ######################## 1) poly_reg ########################
    @st.experimental_memo(suppress_st_warning=True)
    def poly_reg(X_train,X_test,y_train,y_test):
            max_tr_r2_pr, max_te_r2_pr, max_deg=0,0,0
            tr_r2_pr1,te_r2_pr1=[],[] 
            for i in range(1,7):
                    p_reg=LinearRegression()
                    pf1=PolynomialFeatures(degree=i, include_bias=False)
                    X_ptrain=pf1.fit_transform(X_train)
                    X_ptest=pf1.fit_transform(X_test)
                    pr_model=p_reg.fit(X_ptrain,y_train)
                    y_pred_train_pr=pr_model.predict(X_ptrain)
                    y_pred_test_pr=pr_model.predict(X_ptest)

                    tr_r2_pr=r2_score(y_train, y_pred_train_pr)
                    te_r2_pr=r2_score(y_test, y_pred_test_pr)
                    tr_r2_pr1.append(tr_r2_pr)
                    te_r2_pr1.append(te_r2_pr)
                    if max_te_r2_pr<te_r2_pr:
                        max_te_r2_pr=te_r2_pr
                        max_tr_r2_pr=tr_r2_pr
                        y_pred_train_pr_max=y_pred_train_pr
                        y_pred_test_pr_max=y_pred_test_pr
                        max_deg=i
                        pr_model_max=pr_model
                        X_ptest_max=X_ptest
            return pr_model_max,max_deg,y_test,max_tr_r2_pr,max_te_r2_pr,tr_r2_pr1,te_r2_pr1

    ######################## 2) decision_tree ########################
    @st.experimental_memo(suppress_st_warning=True)
    def decision_tree(X_train,X_test,y_train,y_test,n_splits,grid_search_dt):
        if grid_search_dt=='simple':
                parameters={"regressor__splitter":["best"],
                    "regressor__max_depth" : [int(x) for x in np.linspace(2,15,10)],
                  "regressor__min_samples_leaf":[int(x) for x in np.linspace(1,5,3)],
                  "regressor__max_features":["auto"] }
                cv=n_splits
                dt = Pipeline([('scaler',  RobustScaler()),
                    ('regressor', DecisionTreeRegressor(random_state = 0))])

                gs_dt = RandomizedSearchCV(estimator=dt,param_distributions=parameters,cv=cv, n_jobs=-1,random_state=0)
                dt_model=gs_dt.fit(X_train, y_train)
                tuned_dt_model= DecisionTreeRegressor(max_depth=dt_model.best_params_['regressor__max_depth'],max_features=dt_model.best_params_['regressor__max_features'],min_samples_leaf=dt_model.best_params_['regressor__min_samples_leaf'],splitter=dt_model.best_params_['regressor__splitter'],random_state=0)
                tuned_dt_model.fit(X_train,y_train)            

        elif grid_search_dt=='exhaustive':
                parameters={"regressor__splitter":["best","random"],
                    "regressor__max_depth" : [int(x) for x in np.linspace(2,15,12)],
                  "regressor__min_samples_leaf":[int(x) for x in np.linspace(1,5,3)],
                  "regressor__max_features":["auto","log2","sqrt",None] }
                cv=n_splits

                dt = Pipeline([('scaler',  RobustScaler()),
                    ('regressor', DecisionTreeRegressor(random_state = 0))])

                gs_dt = RandomizedSearchCV(estimator=dt,param_distributions=parameters,random_state=0,cv=cv, n_jobs=-1)
                dt_model=gs_dt.fit(X_train, y_train)
                tuned_dt_model= DecisionTreeRegressor(max_depth=dt_model.best_params_['regressor__max_depth'],max_features=dt_model.best_params_['regressor__max_features'],min_samples_leaf=dt_model.best_params_['regressor__min_samples_leaf'],splitter=dt_model.best_params_['regressor__splitter'],random_state=0)
                tuned_dt_model.fit(X_train,y_train)

        y_pred_test_dt=tuned_dt_model.predict(X_test)
        y_pred_train_dt=tuned_dt_model.predict(X_train)

        tr_r2=r2_score(y_train, y_pred_train_dt)
        te_r2=r2_score(y_test, y_pred_test_dt)
        return tuned_dt_model,tr_r2,te_r2,gs_dt.best_params_

    ######################## 3) random_forest ########################
    @st.experimental_memo(suppress_st_warning=True)
    def random_forest(X_train,X_test,y_train,y_test,n_splits,grid_search_rf):
          if grid_search_rf=='simple':
                      grid_rf = {'n_estimators': [int(x) for x in np.linspace(10,100,3)],
                          'criterion': ['friedman_mse'], 
                          'max_features': ["auto"],  
                          'min_samples_split': [int(x) for x in np.linspace(2,4,2)],
                          "bootstrap"    : [False],
                          'min_samples_leaf': [int(x) for x in np.linspace(1,2,2)]}
                      cv=n_splits

          elif grid_search_rf=='exhaustive':
                      grid_rf = {'n_estimators': [int(x) for x in np.linspace(10,100,6)],
                          'criterion': ['friedman_mse', 'squared_error'], 
                          'max_features': ["auto", "sqrt", "log2"],  
                          'min_samples_split': [int(x) for x in np.linspace(2,4,2)],
                          "bootstrap"    : [True, False],
                          'min_samples_leaf': [int(x) for x in np.linspace(1,5,3)]}
                      cv=n_splits

          rf1 = RandomForestRegressor(random_state=0)
          gs_rf = RandomizedSearchCV(rf1, grid_rf,random_state=0, cv=cv, n_jobs=-1)
          try:
              import warnings
              warnings.filterwarnings("error")
              rf_model1=gs_rf.fit(X_train, y_train)
          except:
                  try:
                      rf_model1=gs_rf.fit(X_train, y_train.values.ravel())
                  except:
                      st.write('Error')
          rforest = RandomForestRegressor(n_estimators=gs_rf.best_params_['n_estimators'], criterion = gs_rf.best_params_['criterion'], max_features=gs_rf.best_params_['max_features'], 
          min_samples_split=gs_rf.best_params_['min_samples_split'], min_samples_leaf=gs_rf.best_params_['min_samples_leaf'],bootstrap=gs_rf.best_params_['bootstrap'], random_state=0)
          try:
                import warnings
                warnings.filterwarnings("error")
                rfmodel=rforest.fit(X_train, y_train)
          except:
                try:
                    rfmodel=rforest.fit(X_train, y_train.values.ravel())
                except:
                    st.write('Error')
          y_pred_test_rf = rfmodel.predict(X_test)
          y_pred_train_rf = rfmodel.predict(X_train)
          tr_r2_rf=r2_score(y_train, y_pred_train_rf)
          te_r2_rf=r2_score(y_test, y_pred_test_rf)

          return rfmodel,tr_r2_rf,te_r2_rf,gs_rf.best_params_


    ######################## 4) KNN Regressor ########################
    @st.experimental_memo(suppress_st_warning=True)
    def knnreg(X_train,X_test,y_train,y_test,n_splits):
              cvk=n_splits
              scaler = MinMaxScaler(feature_range=(0, 1))
              params = {'n_neighbors':[int(x) for x in np.linspace(2,20,10)]}
              y_train=pd.DataFrame(y_train)
              x_train_scaled = scaler.fit_transform(X_train)
              X_train = pd.DataFrame(x_train_scaled)

              x_test_scaled = scaler.fit_transform(X_test)
              X_test = pd.DataFrame(x_test_scaled)

              knn = neighbors.KNeighborsRegressor()

              model = RandomizedSearchCV(knn, params,random_state=0, cv=cvk,n_jobs=-1)
              try:
                    model.fit(X_train,y_train)
              except:
                    try:
                        model.fit(X_train,y_train.values.ravel())
                    except:
                        st.write('Error')
              tuned_knn_model=neighbors.KNeighborsRegressor(n_neighbors=model.best_params_['n_neighbors'])
              try:
                      tuned_knn_model.fit(X_train,y_train)
              except:
                    try:
                          tuned_knn_model.fit(X_train,y_train.values.ravel())
                    except:
                          st.write('Error')
              y_pred_test_knn=tuned_knn_model.predict(X_test)
              y_pred_train_knn = tuned_knn_model.predict(X_train)
              tr_r2_knn=r2_score(y_train, y_pred_train_knn)
              te_r2_knn=r2_score(y_test, y_pred_test_knn)

              return tuned_knn_model,tr_r2_knn,te_r2_knn,model.best_params_

    ######################## 5) Lasso Regressor ########################
    @st.experimental_memo(suppress_st_warning=True)
    def lasso(X_train,X_test,y_train,y_test,n_splits):
              cvl=n_splits
              y_train=pd.DataFrame(y_train)
              model = MultiTaskLassoCV(cv=cvl, random_state=0, max_iter=-1,n_jobs=-1)
              model.fit(X_train, y_train)
              tuned_las_model=Lasso(alpha=model.alpha_)
              tuned_las_model.fit(X_train, y_train)
              y_pred_test_las=tuned_las_model.predict(X_test)
              y_pred_train_las = tuned_las_model.predict(X_train)
              tr_r2_las=r2_score(y_train, y_pred_train_las)
              te_r2_las=r2_score(y_test, y_pred_test_las)

              return tuned_las_model,tr_r2_las,te_r2_las,model.alpha_

    ######################## 6) PLS Regressor ########################
    @st.experimental_memo(suppress_st_warning=True)
    def pls(X_train,X_test,y_train,y_test,n_splits):
              cvp=n_splits
              simplefilter(action='ignore', category=FutureWarning)
              pls = PLSRegression()
              params1={'n_components':[int(x) for x in np.linspace(1,10,10)]}
              gs_pls=RandomizedSearchCV(pls, params1, random_state=0,cv = cvp,n_jobs=-1)
              gs_pls.fit(scale(X_train),y_train)
              tuned_pls_model=PLSRegression(n_components=gs_pls.best_params_['n_components'])
              tuned_pls_model.fit(scale(X_train),y_train)
              y_pred_test_pls=tuned_pls_model.predict(scale(X_test))
              y_pred_train_pls = tuned_pls_model.predict(scale(X_train))
              tr_r2_pls=r2_score(y_train, y_pred_train_pls)
              te_r2_pls=r2_score(y_test, y_pred_test_pls)

              return tuned_pls_model,tr_r2_pls,te_r2_pls,gs_pls.best_params_

    ######################## 7) Elastic Net Regressor ########################
    @st.experimental_memo(suppress_st_warning=True)
    def enet(X_train,X_test,y_train,y_test,n_splits):
              cve=n_splits
              enet_model = MultiTaskElasticNetCV(cv=cve,n_jobs=-1).fit(X_train, y_train)
              tuned_enet_model= ElasticNet(alpha =enet_model.alpha_).fit(X_train, y_train)
              y_pred_test_enet=tuned_enet_model.predict(X_test)
              y_pred_train_enet = tuned_enet_model.predict(X_train)
              tr_r2_enet=r2_score(y_train, y_pred_train_enet)
              te_r2_enet=r2_score(y_test, y_pred_test_enet)

              return tuned_enet_model,tr_r2_enet,te_r2_enet,enet_model.alpha_

    ######################################################################################################
        ######################## 1) logistic_regression ########################
    @st.experimental_memo(suppress_st_warning=True)
    def log_reg(X_train,X_test,y_train,y_test,n_splits):
            lmodel = LogisticRegression(max_iter=1000000)
            cv = KFold(n_splits=n_splits)
            space = dict()
            space['solver'] = ['lbfgs', 'liblinear']
            space['C'] = [float(x) for x in np.linspace(0.1,100,6)]
            gs_c = RandomizedSearchCV(lmodel, space, scoring='accuracy', n_jobs=-1, cv=cv,random_state=0)
            model = gs_c.fit(X_train,y_train.values.ravel())
            if model.best_params_['solver']=='liblinear':
                tuned_model = LogisticRegression(solver='liblinear',penalty='l1',C=model.best_params_['C'],max_iter=1000000)
            elif model.best_params_['solver']=='lbfgs':
                tuned_model = LogisticRegression(solver='lbfgs',penalty='l2',C=model.best_params_['C'],max_iter=1000000)
            tuned_model.fit(X_train,y_train.values.ravel())
            y_pred_test_lr=tuned_model.predict(X_test)
            y_pred_train_lr = tuned_model.predict(X_train)

            return tuned_model,y_pred_train_lr,y_pred_test_lr,gs_c.best_params_

    ######################## 2) knn classifier ########################
    @st.experimental_memo(suppress_st_warning=True)
    def knn_cla(X_train,X_test,y_train,y_test,n_splits):
            scaler = MinMaxScaler(feature_range=(0, 1))
            params = {'n_neighbors':[int(x) for x in np.linspace(2,15,13)]}
            y_train=pd.DataFrame(y_train)
            x_train_scaled = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(x_train_scaled)

            x_test_scaled = scaler.fit_transform(X_test)
            X_test = pd.DataFrame(x_test_scaled)

            knn = neighbors.KNeighborsClassifier()
            model = RandomizedSearchCV(knn, params, cv=n_splits,n_jobs=-1,random_state=0)
            try:
                model.fit(X_train,y_train.values.ravel())
            except:
                try:
                    model.fit(X_train,y_train)
                except:
                    st.write('Error')
            tuned_knn_model=neighbors.KNeighborsClassifier(n_neighbors=model.best_params_['n_neighbors'])
            try:
                  tuned_knn_model.fit(X_train,y_train.values.ravel())
            except:
                try:
                      tuned_knn_model.fit(X_train,y_train)
                except:
                      st.write('Error')
            y_pred_test_knn=tuned_knn_model.predict(X_test)
            y_pred_train_knn = tuned_knn_model.predict(X_train)

            return tuned_knn_model,y_pred_train_knn,y_pred_test_knn,scaler,model.best_params_

    ######################## 3) gnb classifier ########################
    @st.experimental_memo(suppress_st_warning=True)
    def gnb_cla(X_train,X_test,y_train,y_test,n_splits):
            gnb = GaussianNB()
            params = {'var_smoothing': np.logspace(0,-9, num=100)}
            gs_nb= RandomizedSearchCV(estimator=gnb,param_distributions=params,cv=n_splits,scoring='accuracy',n_jobs=-1,random_state=0) 
            try:
                gs_nb.fit(X_train,y_train.values.ravel())
            except:
                try:
                    gs_nb.fit(X_train,y_train)
                except:
                    st.write('Error')
            model = GaussianNB(var_smoothing=gs_nb.best_params_['var_smoothing'])
            try:
                tuned_model=model.fit(X_train,y_train.values.ravel())
            except:
                try:
                    tuned_model=model.fit(X_train,y_train)
                except:
                    st.write('Error')
            y_pred_test_gnb=model.predict(X_test)
            y_pred_train_gnb = model.predict(X_train)

            return tuned_model,y_pred_train_gnb,y_pred_test_gnb,gs_nb.best_params_

    ######################## 4) Decision Tree Classifier ########################
    @st.experimental_memo(suppress_st_warning=True)
    def decision_tree_classifier(X_train,X_test,y_train,y_test,n_splits,grid_search_dt):
            if grid_search_dt=='simple':
                    parameters={"splitter":["best"],
                        "max_depth" : [int(x) for x in np.linspace(2,10,3)],
                      "min_samples_leaf":[int(x) for x in np.linspace(1,2,2)],
                      "max_features":["auto",None] }
                    cv=n_splits
            elif grid_search_dt=='exhaustive':
                    parameters={"splitter":["best","random"],
                        "max_depth" : [int(x) for x in np.linspace(2,10,7)],
                      "min_samples_leaf":[int(x) for x in np.linspace(1,3,3)],
                      "max_features":["auto","log2","sqrt",None] }
                    cv=n_splits
            try:
                    dt=DecisionTreeClassifier(random_state = 0)
                    gs_dt = RandomizedSearchCV(dt, param_distributions=parameters,cv=cv, n_jobs=-1,random_state=0)
                    dt_model=gs_dt.fit(X_train, y_train)
                    tuned_dt_model= DecisionTreeClassifier(max_depth=dt_model.best_params_['max_depth'],max_features=dt_model.best_params_['max_features'],min_samples_leaf=dt_model.best_params_['min_samples_leaf'],splitter=dt_model.best_params_['splitter'],random_state=0)
                    tuned_dt_model.fit(X_train,y_train)

                    y_pred_test_dt=tuned_dt_model.predict(X_test)
                    y_pred_train_dt=tuned_dt_model.predict(X_train)
            except:
                st.warning('Error.')
            return tuned_dt_model,y_pred_train_dt,y_pred_test_dt,gs_dt.best_params_

    ######################## 5) Random Forest Classifier ########################
    @st.experimental_memo(suppress_st_warning=True)
    def random_forest_classifier(X_train,X_test,y_train,y_test,n_splits,grid_search_rf):
          if grid_search_rf=='simple':
                  grid_rf = {'n_estimators': [int(x) for x in np.linspace(10,100,3)],
                      'max_features': ["auto"],  
                      'min_samples_split': [int(x) for x in np.linspace(2,3,2)],
                      "bootstrap"    : [False],
                      'min_samples_leaf': [int(x) for x in np.linspace(1,2,2)]}
          elif grid_search_rf=='exhaustive': 
                  grid_rf = {'n_estimators': [int(x) for x in np.linspace(10,200,5)],
                  'criterion': ['gini', 'entropy'], 
                  'max_features': ["auto", "sqrt", "log2"],  
                  'min_samples_split': [int(x) for x in np.linspace(2,3,2)],
                  "bootstrap"    : [True, False],
                  'min_samples_leaf': [int(x) for x in np.linspace(1,3,3)]}
          cv=n_splits
          y_train=pd.DataFrame(y_train)
          rf1 = RandomForestClassifier(random_state=0)
          gs_rf = RandomizedSearchCV(rf1, param_distributions=grid_rf, cv=cv, n_jobs=-1,random_state=0)
          try:
              import warnings
              warnings.filterwarnings("error")
              rf_model1=gs_rf.fit(X_train, y_train)
          except:
                  try:
                      rf_model1=gs_rf.fit(X_train, y_train.values.ravel())
                  except:
                      st.warning('Error')
          rforest = RandomForestClassifier(n_estimators=gs_rf.best_params_['n_estimators'], max_features=gs_rf.best_params_['max_features'], 
          min_samples_split=gs_rf.best_params_['min_samples_split'], min_samples_leaf=gs_rf.best_params_['min_samples_leaf'],bootstrap=gs_rf.best_params_['bootstrap'], random_state=0)
          try:
                rfmodel=rforest.fit(X_train, y_train)
          except:
                try:
                    rfmodel=rforest.fit(X_train, y_train.values.ravel())
                except:
                    st.warning('Error')
          y_pred_test_rf = rfmodel.predict(X_test)
          y_pred_train_rf = rfmodel.predict(X_train)

          return rfmodel,y_pred_train_rf,y_pred_test_rf,gs_rf.best_params_

    ######################## 6) Support Vector Classifier ########################
    @st.experimental_memo(suppress_st_warning=True)
    def svc(X_train,X_test,y_train,y_test,n_splits):
            scaler=MinMaxScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.fit_transform(X_test)
            params={'C':[float(x) for x in np.linspace(0.1,100,6)],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['auto']}
            svc = SVC(probability=True,max_iter=-1)
            model = RandomizedSearchCV(svc, params,cv=n_splits,n_jobs=-1,random_state=0)
            try:
                model.fit(X_train,y_train.values.ravel())
            except:
                try:
                    model.fit(X_train,y_train)
                except:
                    print('Error')
            tuned_svc_model=SVC(probability=True,C=model.best_params_['C'],kernel=model.best_params_['kernel'],gamma=model.best_params_['gamma'],max_iter=-1)
            try:
                tuned_svc_model.fit(X_train,y_train.values.ravel())
            except:
                try:
                      tuned_svc_model.fit(X_train,y_train)
                except:
                      print('Error')
            y_pred_test_svc =tuned_svc_model.predict(X_test)
            y_pred_train_svc = tuned_svc_model.predict(X_train)

            return tuned_svc_model,y_pred_train_svc,y_pred_test_svc,scaler,model.best_params_

    ######################## 8) Extra Trees Classifier ########################
    @st.experimental_memo(suppress_st_warning=True)
    def et_cla(X_train,X_test,y_train,y_test,n_splits,grid_search_et):     
            import warnings
            warnings.filterwarnings("ignore")
            if grid_search_et=='simple':
                params = {'n_estimators':[int(x) for x in np.linspace(100,200,2)],
                'max_depth':[int(x) for x in np.linspace(3,6,2)],'min_samples_split': [int(x) for x in np.linspace(2,3,2)],}

            if grid_search_et=='exhaustive':
                params = {'n_estimators':[int(x) for x in np.linspace(100,500,4)],
                'max_depth':[int(x) for x in np.linspace(1,9,4)],'min_samples_split': [int(x) for x in np.linspace(2,5,3)],}

            gs_et = RandomizedSearchCV(ExtraTreesClassifier(random_state=0),param_distributions=params,cv=n_splits, n_jobs=-1,random_state=0)
            et_model=gs_et.fit(X_train, y_train)

            tuned_et_model= ExtraTreesClassifier(n_estimators=gs_et.best_params_['n_estimators'],
                                          max_depth=gs_et.best_params_['max_depth'],
                                          min_samples_split=gs_et.best_params_['min_samples_split'],                                          
                                          random_state=0)

            tuned_et_model.fit(X_train,y_train)  

            y_pred_test_et=tuned_et_model.predict(X_test)
            y_pred_train_et=tuned_et_model.predict(X_train)

            return tuned_et_model,y_pred_train_et,y_pred_test_et, gs_et.best_params_

    ######################################################################################################

    ######################################################################################################
    if regtype=='Principal Component Analysis':
            n_components=st.sidebar.number_input('n_components',1,10,3)
            X_train_scaled=RobustScaler().fit_transform(X_train)
            pca = PCA(n_components=n_components).fit(X_train_scaled)
            fig = plt.figure(figsize=(9,7))            
            plt.plot(pca.explained_variance_ratio_.cumsum(),lw=2)
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')            
            plt.title('Cumulative explained variance by number of principal components', size=13)
            st.pyplot(plt)
            loadings = pd.DataFrame(
            data=pca.components_.T * np.sqrt(pca.explained_variance_), 
            columns=[f'PC{i}' for i in range(1, n_components+1)],
            index=X_train.columns
            )
            st.write(loadings)
            pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
            pc1_loadings = pc1_loadings.reset_index()
            pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

            pc2_loadings = loadings.sort_values(by='PC2', ascending=False)[['PC2']]
            pc2_loadings = pc2_loadings.reset_index()
            pc2_loadings.columns = ['Attribute', 'CorrelationWithPC2']


            fig = plt.figure(figsize=(8,6))            
            plt.barh(y=pc1_loadings['Attribute'], width=pc1_loadings['CorrelationWithPC1'], color='#097E8B')
            plt.ylabel('Principal Components')
            plt.xlabel('Loading Scores')
            plt.xticks(rotation='vertical')
            plt.title('PCA loading scores (first principal component)', size=13)
            st.pyplot(plt)

            fig = plt.figure(figsize=(8,6))            
            plt.barh(y=pc2_loadings['Attribute'], width=pc2_loadings['CorrelationWithPC2'], color='#057E8B')
            plt.ylabel('Principal Components')
            plt.xlabel('Loading Scores')
            plt.xticks(rotation='vertical')
            plt.title('PCA loading scores (second principal component)', size=13)
            st.pyplot(plt)

    if regtype=='Polynomial Regression':
            rmp=st.radio('Run Model',['n','y'])
            if rmp=='y':
                tuned_modelpr,max_deg2,y_test2,max_tr_r2_pr,max_te_r2_pr,tr_r2_pr1,te_r2_pr1=poly_reg(X_train,X_test,y_train,y_test)
                st.write('Train R2 Score: ', max_tr_r2_pr)
                st.write('Test R2 Score: ', max_te_r2_pr)
                #st.write('---')

                deg = [1,2,3,4,5,6]
                fig = plt.figure(figsize=(9,7))
                plt.plot(deg,tr_r2_pr1,label='Train R2')
                plt.plot(deg,te_r2_pr1,label='Test R2')
                for i in range(1,7):
                      plt.text(i,tr_r2_pr1[i-1],'%.2f'%(tr_r2_pr1[i-1]),horizontalalignment='center',
                      verticalalignment='top')
                for i in range(1,7):
                     plt.text(i,te_r2_pr1[i-1],'%.2f'%(te_r2_pr1[i-1]),horizontalalignment='center',
                     verticalalignment='bottom')
                plt.xlabel('Degree')
                plt.ylabel('Train R2 & Test R2')
                plt.title('Train and Test R2 vs Polynomial Degree')
                plt.legend(loc='best')
                plt.tight_layout()
                st.pyplot(plt)
                count=0
                uploaded_file1 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file1(uploaded_file1):
                        dat=csv.read_csv(uploaded_file1)
                        data2=dat.to_pandas()  
                        return data2
                if uploaded_file1 is not None:
                    X2=read_file1(uploaded_file1)
                    count+=1
                    try:
                        @st.experimental_memo
                        def download_results1(X2,max_deg2,y_test2):
                            p_reg2=LinearRegression()
                            pf2=PolynomialFeatures(degree=max_deg2, include_bias=False)
                            X_p2=pf2.fit_transform(X2)
                            y_pred_pr_final=tuned_modelpr.predict(X_p2)
                            y_pred_pr_final=pd.DataFrame(y_pred_pr_final,columns=y_test2.columns+'_pred')
                            result1=pd.concat([X2,y_pred_pr_final],axis=1)
                            return result1.to_csv().encode('utf-8')

                        result1=download_results1(X2,max_deg2,y_test2)
                        st.success('Success')
                        if result1 is not None:
                            st.download_button(label='Download Predictions',data=result1,file_name='Polynomial_Reg_Predictions.csv',mime='text/csv')       
                    except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')

                else:
                    st.warning('Upload data for predictions')

    if regtype=='Decision Tree':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            grid_search_dt=st.sidebar.radio('Random Search',['simple','exhaustive'])
            rmp1=st.radio('Run Model',['n','y'])
            if rmp1=='y':
                tuned_dt_model,tr_r2,te_r2,best_params=decision_tree(X_train,X_test,y_train,y_test,n_splits,grid_search_dt)
                st.write('Decision Tree Train R2 Score: ', tr_r2)
                st.write('Decision Tree Test R2 Score: ', te_r2)
                st.write('Decision Tree Regressor Best Parameters ',best_params)
                with st.expander('Text Tree'):
                  fname=[]
                  for i in range(len(X_train.columns)):
                       fname.append(X_train.columns[i])
                  text_tree_1 = tree.export_text(tuned_dt_model,feature_names=fname)
                  st.text(text_tree_1)

                count=0
                uploaded_file2 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file2(uploaded_file2):
                    dat=csv.read_csv(uploaded_file2)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file2 is not None:
                     X1=read_file2(uploaded_file2)
                     count+=1
                     try:
                        tuned_dt_modelr=tuned_dt_model
                        @st.experimental_memo
                        def download_results2(X1,y_test):
                              y_pred_dt_final=tuned_dt_modelr.predict(X1)
                              y_pred_dt_final=pd.DataFrame(y_pred_dt_final,columns=y_test.columns+'_pred')
                              result2=pd.concat([X1,y_pred_dt_final],axis=1)
                              return result2.to_csv().encode('utf-8')

                        result2=download_results2(X1,y_test)
                        st.success('Success')
                        if result2 is not None:
                            st.download_button(label='Download Predictions',data=result2,file_name='Decision_Tree_Reg_Predictions.csv',mime='text/csv')       
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Random Forest':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            grid_search_rf=st.sidebar.radio('Random Search',['simple','exhaustive'])
            rmp2=st.radio('Run Model',['n','y'])
            if rmp2=='y':
                tuned_rf_model,tr_r2,te_r2,best_params=random_forest(X_train,X_test,y_train,y_test,n_splits,grid_search_rf)
                st.write('Random Forest Train R2 Score: ', tr_r2)
                st.write('Random Forest Test R2 Score: ', te_r2)
                st.write('Random Forest Regressor Best Parameters ',best_params)

                count=0
                uploaded_file3 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file3(uploaded_file3):
                    dat=csv.read_csv(uploaded_file3)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file3 is not None:
                     X1=read_file3(uploaded_file3)
                     count+=1
                     try:
                        @st.experimental_memo
                        def download_results3(X1,y_test):
                              y_pred_rf_final=tuned_rf_model.predict(X1)
                              y_pred_rf_final=pd.DataFrame(y_pred_rf_final,columns=y_test.columns+'_pred')
                              result3=pd.concat([X1,y_pred_rf_final],axis=1)
                              return result3.to_csv().encode('utf-8')

                        result3=download_results3(X1,y_test)
                        st.success('Success')
                        if result3 is not None:
                            st.download_button(label='Download Predictions',data=result3,file_name='Random_Forest_Reg_Predictions.csv',mime='text/csv')       
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='KNN':                
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp3=st.radio('Run Model',['n','y'])
            if rmp3=='y':    
                tuned_knn_model,tr_r2_knn,te_r2_knn,best_params = knnreg(X_train,X_test,y_train,y_test,n_splits)
                st.write('KNN Train R2 Score: ', tr_r2_knn)
                st.write('KNN Test R2 Score: ', te_r2_knn)
                st.write('KNN Regressor Best Parameters ',best_params)

                count=0
                uploaded_file4 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file4(uploaded_file4):
                    dat=csv.read_csv(uploaded_file4)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file4 is not None:
                    X1=read_file4(uploaded_file4)
                    count+=1
                    try:
                        @st.experimental_memo
                        def download_results4(X1,y_test):
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                X11 = scaler.fit_transform(X1)
                                X11=pd.DataFrame(X11)
                                y_pred_knn_final=tuned_knn_model.predict(X11)
                                y_pred_knn_final=pd.DataFrame(y_pred_knn_final,columns=y_test.columns+'_pred')
                                result4=pd.concat([X1,y_pred_knn_final],axis=1)
                                return result4.to_csv().encode('utf-8')

                        result4=download_results4(X1,y_test)
                        st.success('Success')
                        if result4 is not None:
                            st.download_button(label='Download Predictions',data=result4,file_name='KNN_Reg_Predictions.csv',mime='text/csv')       
                    except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Lasso':                
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp4=st.radio('Run Model',['n','y'])
            if rmp4=='y':    
                tuned_las_model,tr_r2_las,te_r2_las,best_params = lasso(X_train,X_test,y_train,y_test,n_splits)
                st.write('Lasso Train R2 Score: ', tr_r2_las)
                st.write('Lasso Test R2 Score: ', te_r2_las)
                st.write('Lasso Regressor Best Alpha: ',best_params)

                count=0
                uploaded_file5 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file5(uploaded_file5):
                    dat=csv.read_csv(uploaded_file5)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file5 is not None:
                    X1=read_file5(uploaded_file5)
                    count+=1
                    try:
                        @st.experimental_memo
                        def download_results5(X1,y_test):
                                y_pred_las_final=tuned_las_model.predict(X1)
                                y_pred_las_final=pd.DataFrame(y_pred_las_final,columns=y_test.columns+'_pred')
                                result5=pd.concat([X1,y_pred_las_final],axis=1)
                                return result5.to_csv().encode('utf-8')

                        result5=download_results5(X1,y_test)
                        st.success('Success')
                        if result5 is not None:
                            st.download_button(label='Download Predictions',data=result5,file_name='Lasso_Reg_Predictions.csv',mime='text/csv')       
                    except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')  

    if regtype=='PLS':                
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp5=st.radio('Run Model',['n','y'])
            if rmp5=='y':    
                tuned_pls_model,tr_r2_pls,te_r2_pls,best_params = pls(X_train,X_test,y_train,y_test,n_splits)
                st.write('PLS Train R2 Score: ', tr_r2_pls)
                st.write('PLS Test R2 Score: ', te_r2_pls)
                st.write('PLS Regressor Best Parameters ',best_params)

                count=0
                uploaded_file6 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file6(uploaded_file6):
                    dat=csv.read_csv(uploaded_file6)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file6 is not None:
                    X1=read_file6(uploaded_file6)
                    count+=1
                    try:
                        @st.experimental_memo
                        def download_results6(X1,y_test):
                                y_pred_pls_final=tuned_pls_model.predict(scale(X1))
                                y_pred_pls_final=pd.DataFrame(y_pred_pls_final,columns=y_test.columns+'_pred')
                                result6=pd.concat([X1,y_pred_pls_final],axis=1)
                                return result6.to_csv().encode('utf-8')

                        result6=download_results6(X1,y_test)
                        st.success('Success')
                        if result6 is not None:
                            st.download_button(label='Download Predictions',data=result6,file_name='PLS_Reg_Predictions.csv',mime='text/csv')       
                    except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Elastic Net':                
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp6=st.radio('Run Model',['n','y'])
            if rmp6=='y':    
                tuned_enet_model,tr_r2_enet,te_r2_enet,best_params = enet(X_train,X_test,y_train,y_test,n_splits)
                st.write('Elastic Net Train R2 Score: ', tr_r2_enet)
                st.write('Elastic Net Test R2 Score: ', te_r2_enet)
                st.write('Elastic Net Regressor Best Alpha ',best_params)

                count=0
                uploaded_file7 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file7(uploaded_file7):
                    dat=csv.read_csv(uploaded_file7)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file7 is not None:
                    X1=read_file7(uploaded_file7)
                    count+=1
                    try:
                        @st.experimental_memo
                        def download_results7(X1,y_test):
                                y_pred_enet_final=tuned_enet_model.predict(X1)
                                y_pred_enet_final=pd.DataFrame(y_pred_enet_final,columns=y_test.columns+'_pred')
                                result7=pd.concat([X1,y_pred_enet_final],axis=1)
                                return result7.to_csv().encode('utf-8')

                        result7=download_results7(X1,y_test)
                        st.success('Success')
                        if result7 is not None:
                            st.download_button(label='Download Predictions',data=result7,file_name='Elastic_Net_Reg_Predictions.csv',mime='text/csv')       
                    except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    ###
    if regtype=='Logistic Regression':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp11=st.radio('Run Model',['n','y'])
            if rmp11=='y':
                tuned_model,y_pred_train_lr,y_pred_test_lr,best_params=log_reg(X_train,X_test,y_train,y_test,n_splits)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_lr,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_lr,zero_division=0))
                st.write('***Logistic Regression Best Parameters:*** ', best_params)
                g_mean_pred_train_lr=geometric_mean_score(y_train.values.ravel(),y_pred_train_lr,average='weighted')
                g_mean_pred_test_lr=geometric_mean_score(y_test.values.ravel(),y_pred_test_lr,average='weighted')
                st.write('***Logistic Regression Train Geometric Mean:*** ', g_mean_pred_train_lr)
                st.write('***Logistic Regression Test Geometric Mean:*** ', g_mean_pred_test_lr)

                cmat=confusion_matrix(y_test, y_pred_test_lr)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                count=0
                uploaded_file11 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file1(uploaded_file11):
                    dat=csv.read_csv(uploaded_file11)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file11 is not None:
                     X2=read_file1(uploaded_file11)
                     count+=1
                     tuned_modellr,y_test2=tuned_model,y_test
                     try:
                        @st.experimental_memo
                        def download_results1(X2,y_test2):
                              y_pred_lr_final=tuned_modellr.predict(X2)
                              y_pred_lr_final=pd.DataFrame(y_pred_lr_final,columns=y_test2.columns+'_pred')
                              result1=pd.concat([X2,y_pred_lr_final],axis=1)  
                              y_pred_prob_lr_final=tuned_modellr.predict_proba(X2)
                              y_pred_prob_lr_final=pd.DataFrame(y_pred_prob_lr_final,columns=label)                                                                              
                              result11=pd.concat([X2,y_pred_prob_lr_final],axis=1)
                              return result1.to_csv().encode('utf-8'),result11.to_csv().encode('utf-8')

                        result1,result11=download_results1(X2,y_test2)
                        st.success('Success')
                        if result1 is not None and result11 is not None:
                            st.download_button(label='Download Predictions',data=result1,file_name='Logistic_Reg_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result11,file_name='Logistic_Reg_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')


    if regtype=='KNN Classifier':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp22=st.radio('Run Model',['n','y'])
            if rmp22=='y':
                tuned_knn_model,y_pred_train_knn,y_pred_test_knn,scaler,best_params=knn_cla(X_train,X_test,y_train,y_test,n_splits)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_knn,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_knn,zero_division=0))
                st.write('***KNN Classifier Best Parameters:*** ', best_params)
                g_mean_pred_train_knn=geometric_mean_score(y_train.values.ravel(),y_pred_train_knn,average='weighted')
                g_mean_pred_test_knn=geometric_mean_score(y_test.values.ravel(),y_pred_test_knn,average='weighted')
                st.write('***KNN Classifier Train Geometric Mean:*** ', g_mean_pred_train_knn)
                st.write('***KNN Classifier Test Geometric Mean:*** ', g_mean_pred_test_knn)

                cmat=confusion_matrix(y_test, y_pred_test_knn)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                count=0
                uploaded_file2 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file2(uploaded_file2):
                    dat=csv.read_csv(uploaded_file2)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file2 is not None:
                     X2=read_file2(uploaded_file2)
                     count+=1
                     tuned_modellr,y_test2,scaler=tuned_knn_model,y_test,scaler
                     try:
                        @st.experimental_memo
                        def download_results2(X2,y_test2):
                              X22= scaler.fit_transform(X2)
                              y_pred_lr_final=tuned_modellr.predict(X22)
                              y_pred_lr_final=pd.DataFrame(y_pred_lr_final,columns=y_test2.columns+'_pred')
                              result2=pd.concat([X2,y_pred_lr_final],axis=1)
                              y_pred_prob_lr_final=tuned_modellr.predict_proba(X22)
                              y_pred_prob_lr_final=pd.DataFrame(y_pred_prob_lr_final,columns=label)                                                                              
                              result22=pd.concat([X2,y_pred_prob_lr_final],axis=1)
                              return result2.to_csv().encode('utf-8'),result22.to_csv().encode('utf-8')

                        result2,result22=download_results2(X2,y_test2)
                        st.success('Success')
                        if result2 is not None and result22 is not None:
                            st.download_button(label='Download Predictions',data=result2,file_name='KNN_Classifier_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result22,file_name='KNN_Classifier_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Gaussian NB Classifier':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp33=st.radio('Run Model',['n','y'])
            if rmp33=='y':
                tuned_model,y_pred_train_gnb,y_pred_test_gnb,best_params=gnb_cla(X_train,X_test,y_train,y_test,n_splits)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_gnb,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_gnb,zero_division=0))
                st.write('***Gaussian NB Classifier Best Parameters:*** ', best_params)
                g_mean_pred_train_gnb=geometric_mean_score(y_train.values.ravel(),y_pred_train_gnb,average='weighted')
                g_mean_pred_test_gnb=geometric_mean_score(y_test.values.ravel(),y_pred_test_gnb,average='weighted')
                st.write('***Gaussian NB Classifier Train Geometric Mean:*** ', g_mean_pred_train_gnb)
                st.write('***Gaussian NB Classifier Test Geometric Mean:*** ', g_mean_pred_test_gnb)

                cmat=confusion_matrix(y_test, y_pred_test_gnb)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                count=0
                uploaded_file3 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file3(uploaded_file3):
                    dat=csv.read_csv(uploaded_file3)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file3 is not None:
                     X2=read_file3(uploaded_file3)
                     count+=1
                     tuned_modellr,y_test2=tuned_model,y_test
                     try:
                        @st.experimental_memo
                        def download_results3(X2,y_test2):
                              y_pred_lr_final=tuned_modellr.predict(X2)
                              y_pred_lr_final=pd.DataFrame(y_pred_lr_final,columns=y_test2.columns+'_pred')
                              result3=pd.concat([X2,y_pred_lr_final],axis=1)
                              y_pred_prob_lr_final=tuned_modellr.predict_proba(X2)
                              y_pred_prob_lr_final=pd.DataFrame(y_pred_prob_lr_final,columns=label)                                                                              
                              result33=pd.concat([X2,y_pred_prob_lr_final],axis=1)
                              return result3.to_csv().encode('utf-8'),result33.to_csv().encode('utf-8')

                        result3,result33=download_results3(X2,y_test2)
                        st.success('Success')
                        if result3 is not None and result33 is not None:
                            st.download_button(label='Download Predictions',data=result3,file_name='Gaussian_NB_Classifier_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result33,file_name='Gaussian_NB_Classifier_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Decision Tree Classifier':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            grid_search_dt=st.sidebar.radio('Random Search',['simple','exhaustive'])
            rmp44=st.radio('Run Model',['n','y'])
            if rmp44=='y':
                tuned_dt_modelr,y_pred_train_dt,y_pred_test_dt,best_params=decision_tree_classifier(X_train,X_test,y_train,y_test,n_splits,grid_search_dt)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_dt,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_dt,zero_division=0))
                st.write('***Decision Tree Classifier Best Parameters:*** ', best_params)
                g_mean_pred_train_dt=geometric_mean_score(y_train.values.ravel(),y_pred_train_dt,average='weighted')
                g_mean_pred_test_dt=geometric_mean_score(y_test.values.ravel(),y_pred_test_dt,average='weighted')
                st.write('***Decision Tree Classifier Train Geometric Mean:*** ', g_mean_pred_train_dt)
                st.write('***Decision Tree Classifier Test Geometric Mean:*** ', g_mean_pred_test_dt)

                cmat=confusion_matrix(y_test, y_pred_test_dt)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                with st.expander('Text Tree'):
                    fname=[]
                    for i in range(len(X_train.columns)):
                       fname.append(X_train.columns[i])
                    text_tree_1 = tree.export_text(tuned_dt_modelr,feature_names=fname)
                    fig = plt.figure(figsize=(25,20))
                    _ = tree.plot_tree(tuned_dt_modelr,feature_names=fname,filled=True)
                    st.text(text_tree_1)

                count=0
                uploaded_file4 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file4(uploaded_file4):
                    dat=csv.read_csv(uploaded_file4)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file4 is not None:
                     X1=read_file4(uploaded_file4)
                     count+=1
                     try:
                        @st.experimental_memo
                        def download_results4(X1,y_test):
                              y_pred_dt_final=tuned_dt_modelr.predict(X1)
                              y_pred_dt_final=pd.DataFrame(y_pred_dt_final,columns=y_test.columns+'_pred')
                              result4=pd.concat([X1,y_pred_dt_final],axis=1)
                              y_pred_prob_lr_final=tuned_dt_modelr.predict_proba(X1)
                              y_pred_prob_lr_final=pd.DataFrame(y_pred_prob_lr_final,columns=label)                                                                              
                              result44=pd.concat([X1,y_pred_prob_lr_final],axis=1)
                              return result4.to_csv().encode('utf-8'),result44.to_csv().encode('utf-8')

                        result4,result44=download_results4(X1,y_test)
                        st.success('Success')
                        if result4 is not None and result44 is not None:
                            st.download_button(label='Download Predictions',data=result4,file_name='Decision_Tree_Classifier_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result44,file_name='Decision_Tree_Classifier_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Random Forest Classifier':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            grid_search_rf=st.sidebar.radio('Random Search',['simple','exhaustive'])
            rmp55=st.radio('Run Model',['n','y'])
            if rmp55=='y':
                tuned_rf_model,y_pred_train_rf,y_pred_test_rf,best_params=random_forest_classifier(X_train,X_test,y_train,y_test,n_splits,grid_search_rf)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_rf,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_rf,zero_division=0))
                st.write('***Random Forest Classifier Best Parameters:*** ', best_params)
                g_mean_pred_train_rf=geometric_mean_score(y_train.values.ravel(),y_pred_train_rf,average='weighted')
                g_mean_pred_test_rf=geometric_mean_score(y_test.values.ravel(),y_pred_test_rf,average='weighted')
                st.write('***Random Forest Classifier Train Geometric Mean:*** ', g_mean_pred_train_rf)
                st.write('***Random Forest Classifier Test Geometric Mean:*** ', g_mean_pred_test_rf)

                cmat=confusion_matrix(y_test, y_pred_test_rf)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                count=0
                uploaded_file5 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file5(uploaded_file5):
                    dat=csv.read_csv(uploaded_file5)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file5 is not None:
                     X1=read_file5(uploaded_file5)
                     count+=1
                     try:
                        @st.experimental_memo
                        def download_results5(X1,y_test):
                            y_pred_rf_final=tuned_rf_model.predict(X1)
                            y_pred_rf_final=pd.DataFrame(y_pred_rf_final,columns=y_test.columns+'_pred')
                            result5=pd.concat([X1,y_pred_rf_final],axis=1)
                            y_pred_prob_lr_final=tuned_rf_model.predict_proba(X1)
                            y_pred_prob_lr_final=pd.DataFrame(y_pred_prob_lr_final,columns=label)                                                                              
                            result55=pd.concat([X1,y_pred_prob_lr_final],axis=1)
                            return result5.to_csv().encode('utf-8'),result55.to_csv().encode('utf-8')

                        result5,result55=download_results5(X1,y_test)
                        st.success('Success')
                        if result5 is not None and result55 is not None:
                            st.download_button(label='Download Predictions',data=result5,file_name='Random_Forest_Classifier_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result55,file_name='Random_Forest_Classifier_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

    if regtype=='Support Vector Classifier':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            rmp66=st.radio('Run Model',['n','y'])
            if rmp66=='y':
                tuned_modellr,y_pred_train_svc,y_pred_test_svc,scaler,best_params=svc(X_train,X_test,y_train,y_test,n_splits)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_svc,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_svc,zero_division=0))
                st.write('***Support Vector Classifier Best Parameters:*** ', best_params)
                g_mean_pred_train_svc=geometric_mean_score(y_train.values.ravel(),y_pred_train_svc,average='weighted')
                g_mean_pred_test_svc=geometric_mean_score(y_test.values.ravel(),y_pred_test_svc,average='weighted')
                st.write('***Support Vector Classifier Train Geometric Mean:*** ', g_mean_pred_train_svc)
                st.write('***Support Vector Classifier Test Geometric Mean:*** ', g_mean_pred_test_svc)

                cmat=confusion_matrix(y_test, y_pred_test_svc)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                count=0
                uploaded_file6 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file6(uploaded_file6):
                    dat=csv.read_csv(uploaded_file6)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file6 is not None:
                     X2=read_file6(uploaded_file6)
                     count+=1
                     y_test2=y_test
                     try:
                        @st.experimental_memo
                        def download_results6(X2,y_test2):
                              X22= scaler.fit_transform(X2)
                              y_pred_lr_final=tuned_modellr.predict(X22)
                              y_pred_lr_final=pd.DataFrame(y_pred_lr_final,columns=y_test2.columns+'_pred')
                              result6=pd.concat([X2,y_pred_lr_final],axis=1)
                              y_pred_prob_lr_final=tuned_modellr.predict_proba(X22)
                              y_pred_prob_lr_final=pd.DataFrame(y_pred_prob_lr_final,columns=label)                                                                              
                              result66=pd.concat([X2,y_pred_prob_lr_final],axis=1)                        
                              return result6.to_csv().encode('utf-8'),result66.to_csv().encode('utf-8')

                        result6,result66=download_results6(X2,y_test2)
                        st.success('Success')
                        if result6 is not None and result66 is not None:
                            st.download_button(label='Download Predictions',data=result6,file_name='Support_Vector_Classifier_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result66,file_name='Support_Vector_Classifier_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')
                    
    if regtype=='Extra Trees Classifier':
            n_splits=st.sidebar.number_input('n_splits',2,50)
            grid_search_et=st.sidebar.radio('Random Search',['simple','exhaustive'])
            rmp88=st.radio('Run Model',['n','y'])
            if rmp88=='y':
                tuned_et_model,y_pred_train_et,y_pred_test_et,best_params=et_cla(X_train,X_test,y_train,y_test,n_splits,grid_search_et)
                st.text('Classification Report Train Data:\n\n '+classification_report(y_train,y_pred_train_et,zero_division=0))
                st.text('Classification Report Test Data:\n\n '+classification_report(y_test,y_pred_test_et,zero_division=0))
                st.write('***Extra Trees Classifier Best Parameters:*** ', best_params)
                g_mean_pred_train_et=geometric_mean_score(y_train.values.ravel(),y_pred_train_et,average='weighted')
                g_mean_pred_test_et=geometric_mean_score(y_test.values.ravel(),y_pred_test_et,average='weighted')
                st.write('***Extra Trees Classifier Train Geometric Mean:*** ', g_mean_pred_train_et)
                st.write('***Extra Trees Classifier Test Geometric Mean:*** ', g_mean_pred_test_et)

                cmat=confusion_matrix(y_test, y_pred_test_et)
                plt.figure(figsize=(10,10))
                ax = sns.heatmap(cmat/np.sum(cmat), annot=True,fmt='.2%', cmap='Blues')
                ax.set_title('Confusion Matrix\n');
                ax.set_xlabel('Predicted Category')
                ax.set_ylabel('Actual Category ');
                a=[]
                for i in range(len(y_test.columns.sort_values())):
                    a.append(y_test.columns.sort_values()[i])
                label=np.sort(y_test[a[0]].unique())
                ax.xaxis.set_ticklabels(label)
                ax.yaxis.set_ticklabels(label)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(plt)

                count=0
                uploaded_file8 = st.file_uploader("", type=["csv"],key=count)        
                @st.experimental_memo
                def read_file8(uploaded_file8):
                    dat=csv.read_csv(uploaded_file8)
                    data2=dat.to_pandas()  
                    return data2
                if uploaded_file8 is not None:
                     X2=read_file8(uploaded_file8)
                     count+=1
                     y_test2=y_test
                     try:
                        @st.experimental_memo
                        def download_results8(X2,y_test2):
                              y_pred_et_final=tuned_et_model.predict(X2)
                              y_pred_et_final=pd.DataFrame(y_pred_et_final,columns=y_test2.columns+'_pred')
                              result8=pd.concat([X2,y_pred_et_final],axis=1)
                              y_pred_prob_et_final=tuned_et_model.predict_proba(X2)
                              y_pred_prob_et_final=pd.DataFrame(y_pred_prob_et_final,columns=label)                                                                              
                              result88=pd.concat([X2,y_pred_prob_et_final],axis=1)                        
                              return result8.to_csv().encode('utf-8'),result88.to_csv().encode('utf-8')

                        result8,result88=download_results8(X2,y_test2)
                        st.success('Success')
                        if result8 is not None and result88 is not None:
                            st.download_button(label='Download Predictions',data=result8,file_name='Extra_Trees_Classifier_Predictions.csv',mime='text/csv')       
                            st.download_button(label='Download Probability Predictions',data=result88,file_name='Extra_Trees_Classifier_Probability_Predictions.csv',mime='text/csv')                                   
                     except:
                        st.write('Check if the uploaded dataset column names are same as trained model input parameters')
                else:
                    st.warning('Upload data for predictions')

except:
    st.warning('Choose Something')

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

st.write('### **About**')
st.info(
 """
            Created by:
            [Parthasarathy Ramamoorthy](https://www.linkedin.com/in/parthasarathyr97/) (Analytics Specialist @ Premium Peanut LLC)
        """)
