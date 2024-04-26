# Industrial-Copper-Modeling


## About Project


    This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads. The models will utilize advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, identifying the distribution of features, and leveraging  Machine Learning algorithm, to predict the selling price and leads accurately.


In this project i have used two supervised machine learning algorithm to predictthe dependend features,Regression for continues value prediction and Classification for catogorical value prediction

## Regression model :


          The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, outlier detection and handling, handling data in wrong format, identifying the distribution of features, and leveraging  Machine Learning models.

## Classification model :

          Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.


## Work Flow:

          **Import necessery libraries
          **Load the data
          **EDA process to know the information about the data
          **Preprocessing


                      **Handling missing values
                      **Handling the data which are in wrong format
                      **Ckeck the distribution of the data and remove the skewness if its skewed
                      **Handling outliers
                      **Check the correlation of features using heat map

                      
           **Split the features as test and train for ML precessing
           **Select the  Machine Learning  model which should give better perfomance than others
           **Train the model using selected algorithm
           **Evaluation of selected model
           **Tune the model using hyper parameter to get better perfomance,Here using GridsearchCV
           ** Store the model using pickling method(serialising),and can De serialise or load the data when it utalize anywhere
           **Finally deployment done  using streamlit app,In streamlit user can give input and will get selling price and status




Here completed the project  Industrial-Copper-Modeling

created by ARSHINA.P
arshizig7@gmail.com

    
           




                      
          







          
