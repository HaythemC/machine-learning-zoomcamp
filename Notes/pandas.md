DataFrames : table with rows (data) and columns (columns)

`pd.DataFrame(data,columns=columns)` : one way to create the dataframe

`pd.DataFrame(data)` DataFrames can also be created by passing a dictionary containing list of dictionary (data). Pandas use the dictionary key(s) as columns 
`df = pd.read_csv(url, index_col=0)` read csv file from remote location such as github 

`df.head(n=2)` show headers and 2 rows 

Series : every column of df is a serie 

`df.Make` show first column 
`df.Make['Make']` get column named Make
`df[['Make','Model]]` Prodviding a list of column that will be returned 
`df['id'] = [1,2,3,4]` Add new column and populate it with value 1,2,3,4
`del df['id']` Delete a column 

Index 
`df.index` : number of rows from 0 to total number of rows
`df.loc[1]` : return row of index number 1 
`df.index = [a,b,c]` : replace visual index with letters 
`df.iloc[1]` : access positional index instead of visual index 
`df.reset_index(drop=True)` : reset visual index to sequential index - remove old index 

Element-wise operations 
`df['Price'] / 10` : divide elements of column price by 10. * - + are applicable. operators ( >= <= ) are also applicable 

Filtering 

`df[df['year'] >= 2015]` return dataframe where column year is greater then 2015
`df[(df['year'] >= 2015) & df['price'] < 2000 ]` include and operation

String operations 
`df['name'].str.lower` lower case all string in column name 
`df['name'].str.replace(' ','_').str.lower` : replace space with underscore and lower case all string

Summarizing operations 

`df.price.mean()` max , min ..etc also applies
`df.price.describe().round(2)` retrun different value including count, mean, stf, min percentile and max in column price and round to 2 
`df.Make.nunique()` show only unique values in column Make 
`df.horsepower.mode()` Get the most frequent value in the column horsepower
`df['horsepower'] = df['horsepower'].fillna(1)` replace missing value in column horspower with 1

Missing values 
`df.isnull()` : return new df with True where value is missing (NaaN) and False where it is not missing
`df.isnull().sum()` show missing values in each column

Grouping 
`df.groupby('transmission').price.mean()` calculate mean price grouped by transmission

Getting the NumPy arrays 
`df.price.values` return array of values in column price

`df.to_dict(orient='records')` : a list of dictionary 