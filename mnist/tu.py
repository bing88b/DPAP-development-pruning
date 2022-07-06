import numpy as np
import matplotlib.pyplot as plt

x=np.random.rand(100)
maxx=np.percentile(x, 80)
minx=np.min(x)
marge=maxx-minx
xx=(x-minx)/marge
xx=np.clip(xx, 0,1)
xx=xx*2-1
print(xx)
plt.plot(x,xx)
plt.show()