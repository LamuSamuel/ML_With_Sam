# Matplot - Useful for making plots

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)
print(x)
y = np.sin(x)
print(y)
z = np.cos(x)

# # Sine wave
plt.plot(x,y)
plt.show()
#
# # Cosine wave
plt.plot(x,z)
plt.show()
#
# # adding attributes to the graph
plt.plot(x,y)
plt.xlabel('Angle')
plt.ylabel('Sine Value')
plt.title('Sine wave')
plt.show()

# plotting parabola
x = np.linspace(-10,10,20)

y = x**2
plt.plot(x,y,'b*')
plt.xlabel('range')
plt.ylabel('Square of x')
plt.title('Parabola')
plt.show()

# plotting both sine and cosine without y

x = np.linspace(-5,5,50)
plt.plot(x,np.sin(x),'r+')
plt.plot(x,np.cos(x),'b--')
plt.show()

# Bar chat

fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
Languages = ['English','Math','Social','Science','Biology']
Students = [20,30,40,50,70]
axes.bar(Languages,Students)
plt.xlabel('Languages')
plt.ylabel('Students')
plt.show()

# Pie chat
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
Sports = ['Basket Ball','Tennis','Pickle ball','Badminton','Cricket']
players = [17,23,40,22,41]
axes.pie(players,labels=Sports,autopct='%1.1f%%')
plt.show()
 # scatter plot

fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
x = np.linspace(0,10,50)
y = np.sin(x)
z = np.cos(x)
axes.scatter(x,y,c = 'r')
axes.scatter(x,z,c='b')
plt.show()
