import numpy as np
from time import process_time

from numpy.ma.core import subtract
from scipy.odr import exponential

# Numpy --- Numerical Python
## Why Np ?
## Good performance , Memory Efficient , More Math operations and Supports Multi Dimensional Arrays
# usual list

# a quick demonstration of why Np.
# lets go with usual approach
start_time  = process_time()
usual_list = [ i for i in range(10000000) ]
end_time = process_time()
print("Usual load :",end_time - start_time)

# lets now test with Numpy
start_time = process_time()
numpy_list = np.arange(10000000)+5
end_time = process_time()
print("Numpy load :",end_time - start_time)
# your data load with Np should be quicker and faster than the usual approach

array_list = [10,20,30,40,50]

# creating a list with numpy
array_with_numpy = np.array([10,20,30,40,50,60,70])
print(array_with_numpy)
print(type(array_with_numpy))

# creating 1 and 2 dim array with numpy
array_one_dim = np.array([2,3]) # 2 columns
array_two_dim = np.array([[2,3],[5,10]]) # 2 rows and 2 columns
print(array_one_dim)
print(array_two_dim)
print(array_one_dim.shape)
print(array_two_dim.shape)

#  creating datatype with float
array_with_float = np.array([1,2,3,4,5],dtype=float)
print(array_with_float)

# place holders in Numpy
#Creating a numpy array of zero values
array_with_zeros = np.zeros((3,3),dtype=int)
print(array_with_zeros)
# creating arrays with ones
array_with_ones = np.ones((4,4),dtype=int)
print(array_with_ones)
# creating arrays with identity Matrix
array_with_particular_value = np.full((2,3),5)
print(array_with_particular_value)
# creatinng an identity matrix
array_identity = np.eye((4),dtype=int)
print(array_identity)
# creating an array with random values from range 5 to 9
array_random = np.random.randint(5,9,(6,6))
print(array_random)

# creating array of evenly spaced values - should mention the number of values required
array_even_space = np.linspace(10,30,6)
print(array_even_space)

# Creating array of evenly spaced values - should mention step
array_step = np.arange(10,40,5)
print(array_step)

# converting list to numpy
basic_list = [9,8,7,6,5,4]
list_numpy = np.asarray(basic_list)
print(type(list_numpy))

# size of array
size_array = np.array([(10,30,50,60,70,80),(4,5,6,7,8,9)])
print(size_array.size)

# adding,subtracting and multiplying two arrays
list_add_one = [10,20,30,40,50]
list_add_two = [1,2,3,4,5]
added_list = np.add(list_add_one,list_add_two)
subtracted_list = np.subtract(list_add_one,list_add_two)
multiplied_list = np.multiply(list_add_one,list_add_two)
print(added_list,subtracted_list,multiplied_list)

# Manipulating an array
org_array = np.random.randint(1,5,(2,3))
print(org_array)
trans_array = org_array.transpose()
print(trans_array)
