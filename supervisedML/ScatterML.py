import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5,5,10)
y=2*x+1

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")
plt.title( 'Graph: y=2x+1')
plt.grid()
plt.show()