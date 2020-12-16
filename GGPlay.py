# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
from matplotlib import pyplot as plt

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder# creating instance of one-hot-encoder
from sklearn.metrics import mean_squared_error



df = pd.read_csv("GooglePlayApp.csv", sep=",")

#dataset size 8281+ lines

#cleaning data
df = df.dropna() #delete very few lines 

""" # columns
['App', 'Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price',
       'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
       'Android Ver', 'Rating']

""" 

#cleaning review : 
tab = list()
for R in df["Reviews"]:
    R = int(R)
    tab.append(R)
df.Reviews = tab
df.Reviews = df.Reviews.astype(int)
   


#cleaning Price
i = 0
for prices in df["Price"]:
    if '$' in prices:
        df["Price"][i] = prices[1:-1]
    else:
        df["Price"][i] = prices
    i += 1 
df.Price = df.Price.astype(float)
    
#cleaning Installs
tab = list()
for install in df["Installs"]:
    install =install.replace(',','')
    install =install.replace('+','')
    tab.append(install)
df.Installs = tab
df.Installs = df.Installs.astype(int)


#cleaning Android Ver

tab = list()
for Ver in df["Android Ver"]:
    if Ver != "Varies with device":
        Ver = Ver[0]
        tab.append(Ver)
    else: 
        Ver = 100  # I put 100 so our ML model understand that there is problem on Android ver that cause usally bad rating 
        tab.append(Ver)

df = df.dropna() #delete very few lines 
df["Android Ver"] = tab
df["Android Ver"] = df["Android Ver"].astype(int)
    

#cleaning Size : Will be delete cause nobody look at size for a review

tab = list()
for s in df["Size"]:
    if s == "Varies with device":
        s = 0
        tab.append(s)
    else :
        s =s.replace('M','')
        tab.append(s)
df.Size = tab
df.Size = df.Installs.astype(int)


#clean App name : we will count the len 
tab = list()
for app in df["App"]:
    if app != "Varies with device":
        app = len(app)
        tab.append(app)
        
df["App"] = tab

#Cleaning Type L

LabelEnc = LabelEncoder()  
df['Type'] = LabelEnc.fit_transform(df["Type"])


#Cleaning Category L
LabelEnc = LabelEncoder()  
df['Category'] = LabelEnc.fit_transform(df["Category"])


#Cleaning Content Rating L
LabelEnc = LabelEncoder()  
df['Content Rating'] = LabelEnc.fit_transform(df["Content Rating"])


#Cleaning Genres L
LabelEnc = LabelEncoder()  
df['Genres'] = LabelEnc.fit_transform(df["Genres"])


#Cleaning Last Updated Drop cause no interest for ML model
df = df.drop(["Last Updated"],axis = 1)

#Cleaning Current Ver Drop cause no interest for ML model
df = df.drop(["Current Ver"],axis = 1)



#ploting some data analyse
plt.scatter(df["Rating"], df["Price"], c='red', alpha=0.5)
plt.title('Scatter plot Rating vs Price ')
plt.xlabel('Price')
plt.ylabel('Rating')
plt.show()

# ML 
from sklearn.linear_model import LinearRegression


y = df["Rating"] 
X = df.drop(['Rating'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


RegLin = LinearRegression()
RegLin.fit(X_train, y_train)
y_pred = RegLin.predict(X_test)

print("MSE" ,  mean_squared_error(y_test, y_pred))

###############
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset
dataframe = df
# split into input (X) and output (Y) variables
X = X
Y = y
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MS" % (results.mean(), results.std()))