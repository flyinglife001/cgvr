import numpy as np
import matplotlib.pyplot as plt




x1 = np.linspace(-2,1,100)
x = np.linspace(-2,2,100)
y = (1 - x1)*(1 - x1)
plt.plot(x1,y,label='sqhinge')

y = 1 - x1
plt.plot(x1,y,label='hinge')

plt.show()



