import numpy as np

a = np.arange(20)
print(a)
print(a.shape)
print(a[0], a[3])

a = a.reshape(2,10)
print(a.shape)
print(a)

a[0]
a[4]
a[0][4]

a = a.reshape(2,5,2)
a
a.shape

a[0]
a[0][3]
a[0][3][1]

b=np.arange(0,40,2).reshape(4,5)
b

a_python_list =  [2**x for x in range(10)]
a_python_list

c = np.array(a_python_list)
c.shape
c

zero_array = np.zeros(10)
zero_array
zero_array.shape

one_array = np.ones(10)
one_array

empty_array = np.empty(100)
empty_array
empty_array = empty_array.reshape(10,10)
empty_array

lucky_array = np.full((5,5),13)
lucky_array


diagonal_array = np.eye(5)
diagonal_array


random_array = np.random.random(10)
random_array


 
linspace_array = np.linspace(100,200, num=5)
linspace_array

