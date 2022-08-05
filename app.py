import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier


st.sidebar.title("Hyper Parameter Tuning ")
st.sidebar.image('https://i.ytimg.com/vi/ZVR2Way4nwQ/maxresdefault.jpg')

#DEFINIG DATASET
st.sidebar.header("Dataset")
sample = st.sidebar.slider("No. of samples" ,min_value=50, max_value=600 , step = 50)
noise  = st.sidebar.slider("noise" ,min_value=0.0, max_value=1.0)

#Preparing
X, y = make_moons(n_samples=sample, noise=noise)
df = pd.DataFrame(dict(Col1=X[:,0], Col2=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
grouped = df.groupby('label')

#PLOTTING DATASET
a , b = grouped
fig,ax = plt.subplots(figsize=(9,5))
plt.scatter(x = a[1].iloc[:,0].values, y = a[1].iloc[:,1], c ="red",            
            marker ="s", 
            edgecolor ="black",
            s = 50)
plt.scatter(x = b[1].iloc[:,0].values, y = b[1].iloc[:,1], c ="blue",            
            marker ="^", 
            edgecolor ="black",
            s = 50)
plt.xlabel("Col1")
plt.ylabel("Col2")
plt.tight_layout()
st.header("Dataset")
st.pyplot(fig)

#USER MENU
user_menu = st.sidebar.radio(
    "Select Classifier",
    ("Decision Tree" , "Random Forest" )
)

#SPLITING DATASET
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state=0)  

#DECISION TREE HYPER-PARAMETER
if user_menu == "Decision Tree":
    print("Decision Tree")
    st.sidebar.title("Hyper Parameters")
    
    criterion = st.sidebar.selectbox("Select Criteria" , ["gini" , "entropy"])
    splitter = st.sidebar.selectbox("Select Splitter" , ["best" , "random"])

    max_depth = st.sidebar.slider("Max Depth" ,min_value = 0, max_value=100, step=1)
    if max_depth == 0:
        max_depth = None

    max_features = st.sidebar.slider("Features" ,min_value=0, max_value=1, step=1)
    if max_features == 0:
        max_features = None   

    min_sample_leaf = st.sidebar.slider("Min Sample Leaf" ,min_value=1, max_value=100, step=1)

    min_sample_split = st.sidebar.slider("Min Sample Split" ,min_value=2, max_value=100, step=1)  

    max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes" ,min_value=0, max_value=30, step=1)
    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    min_impurity_dec = st.sidebar.slider("Min Impurity Dec" ,min_value=0.0, max_value=2.0)
    
    if st.sidebar.button("Run ALgorithm"):
        #TRAINING         
        classifier= DecisionTreeClassifier(criterion = criterion , splitter = splitter,
                                            max_depth = max_depth , max_features = max_features,
                                            min_samples_leaf = min_sample_leaf , min_samples_split = min_sample_split , 
                                            max_leaf_nodes = max_leaf_nodes , min_impurity_decrease = min_impurity_dec)

        classifier.fit(x_train, y_train)  
        y_pred= classifier.predict(x_test)  
        
        #EVALUATION:
        from sklearn.metrics import accuracy_score
        train = classifier.score(x_train, y_train)*100
        result = accuracy_score (y_pred ,  y_test) *100
        st.header("Results")

        #SHOWING RESULTS
        col1,col2= st.columns(2)
        with col1:
            st.header("Test Accuracy")
            st.title(np.around(result,2))

        with col2:
            st.header("Train Accuracy")
            st.title(np.around(train,2))

        
        #PLOTTING TREE
        original_title = '<p style= "color:Red; font-size: 28px;">Visulaization</p>'
        st.markdown(original_title, unsafe_allow_html=True)

                  
        fig1 = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(classifier,feature_names=["Col1","Col2"], class_names= ["0" , "1" ] ,filled=True )
        st.pyplot(fig1)
              
        

        #PLOTTING DECISION BOUNDARIES      
        fig2,ax = plt.subplots(figsize=(9,5))
        scatter_kwargs = {"s" : 60 , "alpha" : 0.7}
        plot_decision_regions(x_train, y_train,                  
                            clf=classifier,zoom_factor=7,colors = "red,blue" , scatter_kwargs = scatter_kwargs)

        
        plt.legend(loc='upper left')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        plt.tight_layout()
        st.header("Train Data")
        st.pyplot(fig2)

        fig3,ax = plt.subplots(figsize=(9,5))
        scatter_kwargs = {"s" : 60 , "alpha" : 0.7}
        plot_decision_regions(x_test, y_test, clf=classifier,zoom_factor=7,colors = "red,blue" , scatter_kwargs = scatter_kwargs)

        plt.legend(loc='upper left')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        st.header("Test Data")
        st.pyplot(fig3)           
            

        
 #RANDOM FOREST :       
            
if user_menu == "Random Forest":
    st.sidebar.header("Hyper Parameters")

    st.sidebar.header("Random Forest")

    n_estimators = st.sidebar.slider("Estimators" ,min_value = 0, max_value=500, step=1)
    if n_estimators == 0:
        n_estimators = 100
    
    bootstrap= st.sidebar.selectbox("Select Bootstrap" , ["True" , "False"])

    max_samples = st.sidebar.slider("max_samples" ,min_value = 0, max_value=100, step=1)
    if max_samples == 0:
        max_samples = None

    st.sidebar.header("Decision Tree")
    
    criterion = st.sidebar.selectbox("Select Criteria" , ["gini" , "entropy"])
   
    max_depth = st.sidebar.slider("Max Depth" ,min_value = 0, max_value=100, step=1)
    if max_depth == 0:
        max_depth = None

    max_features = st.sidebar.selectbox("Select Features" , ["sqrt" , "log2" , "int"])
    if max_features == "int":
        max_features = st.sidebar.slider("Features" ,min_value=0, max_value=1, step=1)
        if max_features == 0:
            max_features = "sqrt"   

    min_sample_leaf = st.sidebar.slider("Min Sample Leaf" ,min_value=1, max_value=100, step=1)

    min_sample_split = st.sidebar.slider("Min Sample Split" ,min_value=2, max_value=100, step=1)  

    max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes" ,min_value=0, max_value=30, step=1)
    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    min_impurity_dec = st.sidebar.slider("Min Impurity Dec" ,min_value=0.0, max_value=2.0)
    
    if st.sidebar.button("Run ALgorithm"):
        print("Random Forest")

        #TRAINING         
        clf = RandomForestClassifier(n_estimators = n_estimators , bootstrap = bootstrap , 
                                            criterion = criterion ,max_samples = max_samples,
                                            max_depth = max_depth , max_features = max_features,
                                            min_samples_leaf = min_sample_leaf , min_samples_split = min_sample_split , 
                                            max_leaf_nodes = max_leaf_nodes , min_impurity_decrease = min_impurity_dec)

        clf.fit(x_train, y_train)  
        y_pred= clf.predict(x_test)  
        
        #EVALUATION:
        from sklearn.metrics import accuracy_score
        train = clf.score(x_train, y_train)*100
        result = accuracy_score (y_pred ,  y_test) *100
        st.header("Results")

        #SHOWING RESULTS
        col1,col2= st.columns(2)
        with col1:
            st.header("Test Accuracy")
            st.title(np.around(result,2))

        with col2:
            st.header("Train Accuracy")
            st.title(np.around(train,2))

        
        #PLOTTING TREE
        original_title = '<p style= "color:Red; font-size: 28px;">Visulaization</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        

        #PLOTTING DECISION BOUNDARIES      
        fig2,ax = plt.subplots(figsize=(9,5))
        scatter_kwargs = {"s" : 60 , "alpha" : 0.7}
        plot_decision_regions(x_train, y_train,                  
                            clf=clf,zoom_factor=7,colors = "red,blue" , scatter_kwargs = scatter_kwargs)

        
        plt.legend(loc='upper left')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        plt.tight_layout()
        st.header("Train Data")
        st.pyplot(fig2)

        fig3,ax = plt.subplots(figsize=(9,5))
        scatter_kwargs = {"s" : 60 , "alpha" : 0.7}
        plot_decision_regions(x_test, y_test, clf=clf,zoom_factor=7,colors = "red,blue" , scatter_kwargs = scatter_kwargs)

        plt.legend(loc='upper left')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        st.header("Test Data")
        st.pyplot(fig3)


        


