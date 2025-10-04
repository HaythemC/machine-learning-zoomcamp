## Data Preparation 

`!wget $url` Downloading for remote URL 

`df = pd.read_csv('data.csv)` read file with pandas

`df.head()` explore data 

`df['Market Category']` or `df.market_category` : data cleaning

`df.columns` : contains index of columns 

`df.columns = df.columns.str.lower().str.replace(' ','_')` make all columns as lower case and replace space with underscore

`df.dtypes` : return type of each columns. Strings are represented as objects in csv file 

`df.dtypes[df.dtypes == 'object]` return True if type is object 

`df.dtypes[df.dtypes == 'object].index` return columns index 

Cleaning up obj columns (strings) to do data cleaning 
~~~~python
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ','_')
~~~~

## Exploratory data analysis (EDA)

Iterate over each column and identify first unique values and shows total number of unique values
~~~~python
for col in df.columns:
    print(col)
    print(df[col].unique(:5))   #identify first unique values 
    print(df[col].nunique())    #shows total number of unique values

~~~~

Visualizae data 

~~~~python

import matplotlib.pyplot as plt
import seaborn as sns 

%matplotlib inline  #Make sure plots can be displayed in the jupyter notebook
~~~~

`sns.histplot(df.price, bins=50)` show histogram of column price. bins is number of bars (price)

`sns.histplot(df.price[df.price < 1000000], bins=50)` : Long tail distribution is where we see outliers in the price (extremely pricy compared to the rest). With this we can ommit outlier values and focus on where most prices are located 

`np.log([1, 10, 1000, 100000])` : Apply logarthemic to price to get rid of long tail distribution. 
`np.log1p([0, 10, 1000, 100000])` : Adds +1 to all elements of the array since log(0) is not possible

`price_logs = np.log1p(df.price)` : apply log to the price column to get rid of long tail distribution. With this we can have normal distribution (similar to guassian function). which is better to train our model. 

`df.isnull().sum()` Show total Missing values per column

## Setting up the validation framework 

Dataset needs to be split into 3 parts ( train | validation | test ) with (60% | 20% | 20%)

`n = len(df)` : shows total records 
`n_val = int(len(df) * 0.2)` : get 20% of the records for validaition
`n_test = int(len(df) * 0.2)` :  get 20% of the records for test
`n_train = n - n_val - n_test` : get the rest of the train

we need to shuffle data otherwise if it is in order we may have wrong df_train. 
`idx = np.arange(n)` : generate number 
`np.random.seed(2)` : keep same shuffle unified even on other computer
`np.random.suffle(idx)`
`df_train = df.iloc[idx[:n_train]]`
`df_val = df.iloc[idx[n_train:n_train+n_val]]`
`df_test = df.iloc[idx[n_train+n_val:]]`



`df_train = df_train.reset_index(drop=True)` get rid of the shuffled index (do same for test and validation )

get target Y from the 3 different data frames 
`y_train = np.log1p(df_train.price.values)`
`y_val = np.log1p(df_val.price.values)`
`y_test = np.log1p(df_test.price.values)`

delete price (target Y) to avoid creating perfect model by using it 
`del df_train['price']`
`del df_val['price']`
`del df_test['price']`

## Linear regression 

We will use train dataset 

`df_train.iloc[10]` checking row numer 10.

Identify columns to be feeded to the model : Engine_fuel_type, engine_hp amd popularity and put values in an array [453,11,86] : g(xi) ~ y where xi = [453,11,86]

Every feature xi (fuel_type, engine_hp and popularity)  has a weight 
g(xi) = W0 + W1 * xi_1 + W2 * xi_2 + W3 * xi_3
in more simple way : $g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.


Linear regression function template to implement the above g(xi)

~~~~python
w0 = 7.17 # if we don't the features of the car, what should we predict as price for random car
w = [0.01, 0.04, 0.002] # weighting the feature 

def g(xi):
    n = len(xi)

    prediction = w0 

    for j in range(n):
        prediction = prediction + w[j] * xi[j]
    return prediction
~~~~

`np.expm1(prediction)` prediction is a logarithmic value - we need to undo the logarithmic using exponoential square -1 to find the actual price

## Linear regression vector form 

$g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

The above equation can be changed into multipling the transpose of vector of features by vector of weights as implemented below 

W = [ w0 , w1, w2 .. ]

Xi = [ Xi0 , Xi1 , Xi2 ...] where xi0 = 1 
this help us to do dot product including w0 

Xi(Transpose). w = $g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

~~~~python

def dot(xi,w):
    n = len(xi)

    res = 0.0
    for j in range(n):
        res = res +xi[j] * w[j]
    return res
~~~~

`w_new = w[0] + w` where w[0] is the w0 = 7.17 and w is the vector of weights w = [0.01, 0.04, 0.002] 

~~~~python
def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new) # we can use numpy dot function
~~~~

~~~~python
x1 = [1, 148, 11, 86] # Feature value 1 
x2 = [1, 132, 11, 1385] # Feature value 2
x3 = [1, 500, 11, 2000]

X = [x1, x2, x3]
np.array(X) 

X.dot(w_new)
~~~~

## Training a linear regression model 

We try to answer here - how to we obtain the weight vector ? 

