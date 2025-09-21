01-Intro
1.7 Introduction to NumPy

Creating array
`np.zeros(5)` : Create an array of 5 elements. Every element is equal to 0
`np.ones(5)` :  Create an array of 5 elements. Every element is equal to 1
`np.full(5, 1.5)` : Create an array of 5 elements. Every element is equal to 1.5

`np.array([1, 2, 5, 12])` : Create an array based on the provided list
`np[2]` : access the 3rd element of an array 

`np.arange(10)` : Create an array. Element are incremental from 0 to 9.
`np.arange(3, 10)` : Create an array. Element are incremental starting from 3. 

`np.linspace(0, 1, 11)` : Create an array of size 11. element of the array ranges from 0 to 1 (0, 0.1, 0.2 ..etc, 1 )

Multi-dimentional array

`np.zeros(5, 2)` : Create an array of 5 rows and 2 columns 
`n = np.array([1,2,4],[3,6,7])` : create an array based on list of list. Create 2 rows and 3 column matrice.
`n[0,1]` access element located in the first row and in the 2nd column (2)
`n[:,1]` get all the rows but only for column indexed with number 1 
`n[0]` get first row 

Randomly generated arrays 

`np.random.rand(5,2)` generate an array of 5 rows and 2 column. Elements are random number. 
`100 * np.random.rand(5,2)` Multiply element value by 100
`np.random.seed(2)` random values are pseudo-random. selecting the seed will ensure that we get the same numbers on every run. 
`np.random.randn(5,2)`
`np.random.randint(low=0, high=100, size=(5, 2))` Generate matrice of 5 rows and 2 column with random element value between 0 and 99

Element-wise operations
`a = np.arange(10)` `a + 1` : adds one to every element of the array. - / * are also supported. 
`a + b` : Sum of elements 

Element-wise comparation
`a >= 2` : return array of element False or True if element is greater or equal to 2
`a > b` : compare elements of two arrays 
`a[ a > b]` : return element of array a where condition a > b is True

Summarizing operations 
`a.min()` : return min of all elmeent of the array a. mean, std, max ..etc also applies. 

[Nmupy Cheat Sheet](https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python)