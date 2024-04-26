#.........................................................Industrial-Copper-Modeling...................................
#Import useful libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder 


#streamlit background settings
st.set_page_config(layout='wide')
st.title(":violet[Industrial Copper Modeling ]")
tab1,tab2=st.tabs(["HOME","APPLICATION"])
with tab1:
    st.text_area(":rainbow[ABOUT PROJECT]:",
                            "The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions.So This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification.")  
    
    st.text_area(":rainbow[WHAT IS STREAMLIT USED FOR]",
                 "Streamlit is a promising open-source Python library, which enables developers to build attractive user interfaces in no time. Streamlit is the easiest way especially for people with no front-end knowledge to put their code into a web application,Here i developed streamlit which allows the user to give input and get selling price and status of  copper Industry data ")

    st.text_area(":rainbow[Tools Used For this Project]","Python,VSCODE,Github,Streamlit")

    st.text_area(":rainbow[Project created by]","ARSHINA.P,     contact:arshizig7@gmail.com")



try:

        with tab2:
           select = st.radio("select one",(":green[PREDICT SELLING PRICE]",":green[PREDICT STATUS]"))
#Selling price prediction
           if select==":green[PREDICT SELLING PRICE]":
                
                col1,col2=st.columns(2)
                with col1:
                  

                
                  #To getting input from user
                        status=st.selectbox(":rainbow[SELECT STATUS]",("0","1","2","3","4","5","6","7","8"))
                        st.write('Draft:0','Lost:1','Not lost for AM:2',"Offerable:3","Offered:4","Revised:5",'To be approved:6','Won:7',"Wonderful:8")
                        status=int(status)
                      


                        country=st.selectbox(":rainbow[SELECT COUNTRY]",("28","25","30","32","38","78","27","77","113","79","26","39","40","84","80","107","89"))
                        country=float(country)

                        item_type=st.selectbox(":rainbow[SELECT ITEM_TYPE]",("0","1","2","3","4","5","6"))
                        st.write('IPL:0', 'Others:1', 'PL:2', 'S:3', 'SLAWR:4', 'W:5', 'WI:6')
                        item_type=int(item_type)
                        

                        application=st.selectbox(":rainbow[SELECT APPLICATION]",("10","41","28","59","15","4","38","56","42","26","27","19","20","66","29","22","40","25","67","79","3","99","2","5","39","69","70","65","58","68"))
                        application=float(application)

                        product_ref=st.selectbox(":rainbow[SELECT PRODUCT_REFERENCE]",("1670798778","1668701718","628377","640665","611993","1668701376","164141591","1671863738","1332077137","640405","1693867550","1665572374","1282007633","1668701698","628117","1690738206","628112","640400","1671876026","164336407","164337175","1668701725","1665572032","611728","1721130331","1693867563","611733","1690738219","1722207579","929423819","1665584320","1665584662","1665584642")) 
                      

                        item_date_year=st.selectbox(":rainbow[SELECT ITEM_DATE_YEAR]",("2021","2020"))
                        item_date_year=int(item_date_year)

                        item_date_month=st.selectbox(":rainbow[SELECT ITEM_DATE_MONTH]",("4","12","3","2","1","11","10","9","8","7"))
                        item_date_month=int(item_date_month)

                        item_date_day=st.selectbox(":rainbow[SELECT ITEM_DATE_DAY]",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
                        item_date_day=int(item_date_day)

                        delivery_date_year=st.selectbox(":rainbow[SELECT DELIVERY_DATE_YEAR]",("2022","2021","2020","2019"))
                        delivery_date_year=int(delivery_date_year)

                        delivery_date_month=st.selectbox(":rainbow[SELECT DELIVERY_DATE_MONTH]",("1","2","3","4","5","6","7","8","9","10","11","12"))
                        delivery_date_month=int(delivery_date_month)

                        delivery_date_day=st.selectbox(":rainbow[SELECT DELIVERY_DATE_DAY]",("1"))
                        delivery_date_day=int(delivery_date_day)

                with col2:
                      quantity_tons=st.text_input(":rainbow[ENTER QUANTITY TON](Min: 611728  & max: 1722207579)")
                    

                      customer=st.text_input(":rainbow[ENTER CUSTOMER] (MIN: 12458.0 & MAX: 2147483647.0)")
                      

                      thickness= st.text_input(":rainbow[ENTER  THICKNESS] (MIN :0.18 & MAX: 2500.0 )")
                      

                      width=st.text_input(":rainbow[ENTER WIDTH ](MIN:1.0 & MAX: 2990.0)")
                      width=float(width)
                      button_R=st.button(":rainbow[PREDICT SELLING PRICE]")
                      if button_R:

                          data={"country":country,"status":status,"item type":item_type,"application":application,"width":width,"quantity_tons_log":quantity_tons,"customer_log":customer,
                            "thickness_log":thickness,"product_ref_log":product_ref,"item_date_year":item_date_year,
                            "item_date_month":item_date_month,"item_date_day":item_date_day,"delivery_date_year":delivery_date_year,
                            "delivery_date_month":delivery_date_month, "delivery_date_day":delivery_date_day}
                          
                    
                          #store all the input as dataframe
                          df=pd.DataFrame(data,index=[1])

                          #convert to log values of input
                          df["customer_log"]=np.log(float(df["customer_log"]))
                          df["quantity_tons_log"]=np.log(float(df["quantity_tons_log"]))
                          df["thickness_log"]=np.log(float(df["thickness_log"]))
                          df["product_ref_log"]=np.log(float(df["product_ref_log"]))
                    


                    
                          #To predict selling price
                          #De serialing of stored model
                          with open("Regression_model.pk1","rb") as f3:
                                                  R_model=pickle.load(f3)

                          with open("Scalar.pk1","rb") as f4:
                                          scalar=pickle.load(f4)
                          new_data1=scalar.transform(df)
                          y_predict_reg=R_model.predict(new_data1)
                          
                          st.write("### :violet[SELLING PRICE IS]")
                          st.write(np.exp(y_predict_reg))



    
#status prediction



           if select==":green[PREDICT STATUS]": 
         
                col3,col4=st.columns(2)
                with col3:
             
                  #Getting input from user for status prediction
                        country=st.selectbox(":rainbow[SELECT COUNTRY]",("28","25","30","32","38","78","27","77","113","79","26","39","40","84","80","107","89"))
                        country=float(country)

                        item_type=st.selectbox(":rainbow[SELECT ITEM_TYPE]",("0","1","2","3","4","5","6"))
                        st.write('IPL:0', 'Others:1', 'PL:2', 'S:3', 'SLAWR:4', 'W:5', 'WI:6')
                        item_type=int(item_type)
                        

                        application=st.selectbox(":rainbow[SELECT APPLICATION]",("10","41","28","59","15","4","38","56","42","26","27","19","20","66","29","22","40","25","67","79","3","99","2","5","39","69","70","65","58","68"))
                        application=float(application)

                        product_ref=st.selectbox(":rainbow[SELECT PRODUCT_REFERENCE]",("1670798778","1668701718","628377","640665","611993","1668701376","164141591","1671863738","1332077137","640405","1693867550","1665572374","1282007633","1668701698","628117","1690738206","628112","640400","1671876026","164336407","164337175","1668701725","1665572032","611728","1721130331","1693867563","611733","1690738219","1722207579","929423819","1665584320","1665584662","1665584642")) 
                      

                        item_date_year=st.selectbox(":rainbow[SELECT ITEM_DATE_YEAR]",("2021","2020"))
                        item_date_year=int(item_date_year)

                        item_date_month=st.selectbox(":rainbow[SELECT ITEM_DATE_MONTH]",("4","12","3","2","1","11","10","9","8","7"))
                        item_date_month=int(item_date_month)

                        item_date_day=st.selectbox(":rainbow[SELECT ITEM_DATE_DAY]",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
                        item_date_day=int(item_date_day)

                        delivery_date_year=st.selectbox(":rainbow[SELECT DELIVERY_DATE_YEAR]",("2022","2021","2020","2019"))
                        delivery_date_year=int(delivery_date_year)

                        delivery_date_month=st.selectbox(":rainbow[SELECT DELIVERY_DATE_MONTH]",("1","2","3","4","5","6","7","8","9","10","11","12"))
                        delivery_date_month=int(delivery_date_month)

                        delivery_date_day=st.selectbox(":rainbow[SELECT DELIVERY_DATE_DAY]",("1"))
                        delivery_date_day=int(delivery_date_day)

                with col4:
                      selling_price=st.text_input(":rainbow[ENTER SELLING PRICE](Min: 1.0  & max: 100001015.0)")


                      quantity_tons=st.text_input(":rainbow[ENTER QUANTITY TON](Min: 611728  & max: 1722207579)")
                    

                      customer=st.text_input(":rainbow[ENTER CUSTOMER] (MIN: 12458.0 & MAX: 2147483647.0)")
                      

                      thickness= st.text_input(":rainbow[ENTER  THICKNESS] (MIN :0.18 & MAX: 2500.0 )")
                      

                      width=st.text_input(":rainbow[ENTER WIDTH ](MIN:1.0 & MAX: 2990.0)")
                      
                      width=float(width)
                      button_C=st.button(":rainbow[PREDICT STATUS]")
                      if button_C:
      

                            data={"country":country,"item type":item_type,"application":application,"width":width,"quantity_tons_log":quantity_tons,"selling_price_log":selling_price,"customer_log":customer,
                              "thickness_log":thickness,"product_ref_log":product_ref,"item_date_year":item_date_year,
                              "item_date_month":item_date_month,"item_date_day":item_date_day,"delivery_date_year":delivery_date_year,
                              "delivery_date_month":delivery_date_month, "delivery_date_day":delivery_date_day}

                      
                      
                            df=pd.DataFrame(data,index=[1])
                            df["customer_log"]=np.log(float(df["customer_log"]))
                            df["quantity_tons_log"]=np.log(float(df["quantity_tons_log"]))
                            df["thickness_log"]=np.log(float(df["thickness_log"]))
                            df["product_ref_log"]=np.log(float(df["product_ref_log"]))
                            df["selling_price_log"]=np.log(float(df["selling_price_log"]))
                    
                      
                      
                      
                    
                    
                            #Classification model to predict status
                            #De serialing of stored model
                            with open("classification_model.pk1","rb") as f_:
                                                            C_model=pickle.load(f_)

                            with open("Scalar_C.pk1","rb") as f4:
                                                    scalar_=pickle.load(f4)
                            new_data=scalar_.transform(df)
                            y_predict_class=C_model.predict(new_data)
                                    
                                    
                            #button_C=st.button(":rainbow[PREDICT STATUS]")
                            
                            
                            if y_predict_class==1:
                                                  st.write("### :green[STATUS IS WON]")
                            else:
                                                    st.write("### :red[STATUS IS LOST]")

except:
      pass

            #Here completed te project Industrial-Copper-Modeling.