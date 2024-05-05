<H3>ENTER YOUR NAME</H3>Elamaran S E
<H3>ENTER YOUR REGISTER NO.</H3>212222230036
<H3>EX. NO.1
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/Churn_Modelling.csv")
print(df)
df.head()
```
```
X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)
```
```
print(df.isnull().sum())

df.duplicated()

print(df['HasCrCard'].describe())
```
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
```
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))

print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
![308941776-80aeba64-938f-4126-8e59-0372ca22733e](https://github.com/elamarannn/Ex-1-NN/assets/113497531/24f1513f-2129-4d40-bf29-cbb382570876)
![308941792-e52b8ada-b2f9-4d34-831e-e765b60ab470](https://github.com/elamarannn/Ex-1-NN/assets/113497531/8e4c3245-fdfe-473b-b970-a8a2dcc944bb)
![308941800-54378abb-a899-4f16-ab3b-082706e6dead](https://github.com/elamarannn/Ex-1-NN/assets/113497531/8d49ec2f-ff5c-428f-bdac-75b6ea544327)
![308941806-8e2c80b5-a849-4dcf-bd33-c8af8a348302](https://github.com/elamarannn/Ex-1-NN/assets/113497531/a4d05e21-9c73-4d6c-abfb-6f90396b11b6)
![308941812-52bf22bc-ddf7-4f42-adee-1c7834dacc48](https://github.com/elamarannn/Ex-1-NN/assets/113497531/ebf8c2f6-9d19-45cc-90b9-a072a530b230)
![308941817-15277847-3a88-4c93-8542-8964d2668904](https://github.com/elamarannn/Ex-1-NN/assets/113497531/cc38a519-198c-4179-b91a-974b1bdb5198)
![308941824-b8dba549-e6af-46bd-b731-6ea11c9fb912](https://github.com/elamarannn/Ex-1-NN/assets/113497531/f3dfe2d2-55bf-45a7-bf56-4654e852dcee)
![308941830-8b3c6318-1a99-4d9c-ac29-784750bc3c96](https://github.com/elamarannn/Ex-1-NN/assets/113497531/25b6cb0b-d51d-4acf-b05b-141ae77e618e)
![308941839-d9d04b8d-f22f-44df-806a-3db0e327e809](https://github.com/elamarannn/Ex-1-NN/assets/113497531/52d522b4-541c-459c-9418-03016c466965)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


