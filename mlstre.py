import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import gc


# Load the dataset
dataset = pd.read_csv("credit card clients.csv")

# Perform data preprocessing steps here
# ...
# Renaming the columns in the dataset
dataset = dataset.rename(columns={'default payment next month':'def_pay','PAY_0':'PAY_1'})



#Dropping unwanted Column, the ID column has unique values it has nothing to do with the output column
dataset.drop('ID', axis=1, inplace=True)

#1--> Male
#2--> Female
dataset['SEX'].value_counts(dropna=False)

dataset['SEX'] = dataset['SEX'].map({1:'Male',2:'Female'})

#1-->graduate school,
#2-->university,
#3-->high school,
#4-->others,
#5-->unknown,
#6-->unknown
dataset['EDUCATION'].value_counts(dropna=False)

dataset['EDUCATION'] = dataset['EDUCATION'].map({1:'Graduate_School',2:'University',3:'High_School',4:'Others',5:'Unknown',6:'Unknown'})

#1-->married,
#2-->single,
#3-->others
dataset['MARRIAGE'].value_counts(dropna=False)

dataset['MARRIAGE'] = dataset['MARRIAGE'].map({1:'Married',2:'Single',3:'Others'})

#1--> Defaulter
#0--> Non Defaulter
dataset['def_pay'].value_counts(dropna=False)

dataset['def_pay'] = dataset['def_pay'].map({0:'Non_Defaulter',1:'Defaulter'})

dataset.head()


# Converting categorical columns into Numerical columns

dataset['SEX'] = dataset['SEX'].map({'Male':1,'Female':2})

dataset['EDUCATION'] = dataset['EDUCATION'].map({'Graduate_School':1,'University':2,'High_School':3,'Others':4,'Unknown':5,'Unknown':6})

dataset['MARRIAGE'] = dataset['MARRIAGE'].map({'Married':1,'Single':2,'Others':3})

dataset['def_pay'] = dataset['def_pay'].map({'Non_Defaulter':0,'Defaulter':1})

dataset.head()




# EDUCATION has category 5 and 6 that are unlabelled,5 and 6 (label unknown) in EDUCATION can also be put in a 'Other' category (thus 4)
fil = (dataset.EDUCATION == 5) | (dataset.EDUCATION == 6) | (dataset.EDUCATION == 0)
dataset.loc[fil, 'EDUCATION'] = 4
dataset.EDUCATION.value_counts()



fil = (dataset.PAY_1 == -1) | (dataset.PAY_1==-2)
dataset.loc[fil,'PAY_1']=0
dataset.PAY_1.value_counts()
fil = (dataset.PAY_2 == -1) | (dataset.PAY_2==-2)
dataset.loc[fil,'PAY_2']=0
dataset.PAY_2.value_counts()
fil = (dataset.PAY_3 == -1) | (dataset.PAY_3==-2)
dataset.loc[fil,'PAY_3']=0
dataset.PAY_3.value_counts()
fil = (dataset.PAY_4 == -1) | (dataset.PAY_4==-2)
dataset.loc[fil,'PAY_4']=0
dataset.PAY_4.value_counts()
fil = (dataset.PAY_5 == -1) | (dataset.PAY_5==-2)
dataset.loc[fil,'PAY_5']=0
dataset.PAY_5.value_counts()
fil = (dataset.PAY_6 == -1) | (dataset.PAY_6==-2)
dataset.loc[fil,'PAY_6']=0
dataset.PAY_6.value_counts()


#Converting column names into lower
dataset.columns = dataset.columns.map(str.lower)



col_to_norm = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4',
       'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3',
       'pay_amt4', 'pay_amt5', 'pay_amt6']
dataset[col_to_norm] = dataset[col_to_norm].apply(lambda x : (x-np.mean(x))/np.std(x))

X = dataset.drop(columns = 'def_pay', axis=1)
Y = dataset['def_pay']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data

Y = dataset['def_pay']

print(X)

print(Y)



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1)


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)



# XGBoost Model
def train_xgboost(predictiontime):
	xgb = XGBClassifier(    n_estimators: 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8)
	xgb.fit(X_train, Y_train)
	y_pred =xgb.predict(X_test)
	from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	roc=roc_auc_score(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)	
	prec = precision_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)	
	f1 = f1_score(y_test, y_pred)

	results = pd.DataFrame([['XGBoost', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
	input_data_as_numpy_array = np.asarray(predictiontime)
	#st.write(input_data_as_numpy_array)

#reshape the array as we are predicting for one instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
	std_data = scaler.transform(input_data_reshaped)
	print(std_data)

	prediction = xgb.predict(std_data)
	print(prediction)

	if (prediction[0] == 0):
    		st.write("The person is not default")
	else:
    		st.write("The person is default")
	return results

# Logistic Regression Model
def train_logistic_regression(predictiontime):
	logmodel = LogisticRegression(random_state=1)
	logmodel.fit(X_train, Y_train)
	y_pred =logmodel.predict(X_test)
	from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	roc=roc_auc_score(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)	
	prec = precision_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)	
	f1 = f1_score(y_test, y_pred)

	results = pd.DataFrame([['Logistic Regression', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
	input_data_as_numpy_array = np.asarray(predictiontime)
	#st.write(input_data_as_numpy_array)

#reshape the array as we are predicting for one instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
	std_data = scaler.transform(input_data_reshaped)
	print(std_data)

	prediction = logmodel.predict(std_data)
	print(prediction)

	if (prediction[0] == 0):
    		st.write("The person is not default")
	else:
    		st.write("The person is default")
	return results

# Support Vector Machine Model
def train_support_vector_machine(predictiontime):
	svm = SVC(kernel='rbf', random_state=0)
	svm.fit(X_train, Y_train)
	y_pred =svm.predict(X_test)
	from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	roc=roc_auc_score(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)	
	prec = precision_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)	
	f1 = f1_score(y_test, y_pred)

	results = pd.DataFrame([['Support Vector Machine', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
	input_data_as_numpy_array = np.asarray(predictiontime)
	#st.write(input_data_as_numpy_array)

#reshape the array as we are predicting for one instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
	std_data = scaler.transform(input_data_reshaped)
	print(std_data)

	prediction = svm.predict(std_data)
	print(prediction)

	if (prediction[0] == 0):
    		st.write("The person is not default")
	else:
    		st.write("The person is default")
	return results

# Decision Tree Classifier Model
def train_decision_tree(predictiontime):
	dct = DecisionTreeClassifier(criterion='entropy', random_state=0)
	dct.fit(X_train, Y_train)
	y_pred =dct.predict(X_test)
	from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	roc=roc_auc_score(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)	
	prec = precision_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)	
	f1 = f1_score(y_test, y_pred)

	results = pd.DataFrame([['Decision Tree', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
	input_data_as_numpy_array = np.asarray(predictiontime)
	#st.write(input_data_as_numpy_array)

#reshape the array as we are predicting for one instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
	std_data = scaler.transform(input_data_reshaped)
	print(std_data)

	prediction = dct.predict(std_data)
	print(prediction)

	if (prediction[0] == 0):
    		st.write("The person is not default")
	else:
    		st.write("The person is default")
	return results


# Random Forest Classifier Model
def train_random_forest(predictiontime):
	rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
	rfc.fit(X_train, Y_train)
	y_pred =rfc.predict(X_test)
	from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
	roc=roc_auc_score(y_test, y_pred)
	acc = accuracy_score(y_test, y_pred)	
	prec = precision_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)	
	f1 = f1_score(y_test, y_pred)

	results = pd.DataFrame([['Random Forest', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
	input_data_as_numpy_array = np.asarray(predictiontime)
	#st.write(input_data_as_numpy_array)

#reshape the array as we are predicting for one instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
	std_data = scaler.transform(input_data_reshaped)
	print(std_data)

	prediction = rfc.predict(std_data)
	print(prediction)

	if (prediction[0] == 0):
    		st.write("The person is not default")
	else:
    		st.write("The person is default")
	return results

def return_head(filename):
    df=pd.read_csv(filename)
    head=df.head(30).reset_index()
    del df
    gc.collect()
    return head

st.image('Banner.png')
st.markdown('## Business Problem')
st.markdown('* The problem is of risk modelling.')
st.markdown('* Given the data of a client we have to predict if he/she will be defaulter in paying credit card bill Next Month.')

def file_selector(folder_path='.'):
	st.title("Credit Card Default Prediction")
	filenames = os.listdir(folder_path)
	filenames.sort()
	selected_filename = st.selectbox('Select correct or error data', filenames)
	return os.path.join(folder_path, selected_filename)


filename = file_selector(folder_path='data')
error_flag=0
df=pd.read_csv(filename)


if error_flag==0:
	df_head=return_head(filename)
	st.write("Test Client's data")
	df_head=df_head.drop(['index'], axis=1)
	st.dataframe(df_head)
	option = st.selectbox("Select the Name",(df_head['Name'].values))
	test_point=df_head[df_head['Name']==option]
	df=test_point.drop(['Name','default payment next month'], axis=1)
	st.write(df)




def main():
	st.subheader("Machine Learning Models")
	model_names = ['XGBoost', 'Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest']
	selected_model = st.selectbox("Select a model", model_names)
	predictiontime=df
    # input_data = [(140000,1,2,1,41,0,0,0,0,0,0,138325,137142,139110,138262,49675,46121,6000,7000,4228,1505,2000,2000),
    #         (140000,1,2,1,41,0,0,0,0,0,0,138325,137142,139110,138262,49675,46121,6000,7000,4228,1505,2000,2000),
    #          (140000,1,2,1,41,0,0,0,0,0,0,138325,137142,139110,138262,49675,46121,6000,7000,4228,1505,2000,2000)]
    # predictiontime = st.selectbox("Select a inputdata", input_data)
    
	if st.button("Train and Predict"):
		if selected_model == 'XGBoost':
			prediction = train_xgboost(predictiontime)
		elif selected_model == 'Logistic Regression':
			prediction = train_logistic_regression(predictiontime)
		elif selected_model == 'Support Vector Machine':
			prediction = train_support_vector_machine(predictiontime)
		elif selected_model == 'Decision Tree':
			prediction = train_decision_tree(predictiontime)
		elif selected_model == 'Random Forest':
			prediction = train_random_forest(predictiontime)

		st.subheader("Prediction Results")
		st.write(prediction)



if __name__ == '__main__':
    main()
