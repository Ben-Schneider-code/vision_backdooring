import matplotlib.pyplot as plt
import numpy as np


data = [.5825, .8041, .806, .8089]
labels = ['50', '75', '100', '150']

#convert data to % from proportion
plt.bar(labels, np.array(data)*100)
plt.xlabel('Number of poisons [thousands]')
plt.ylabel('Acc (%)')
plt.title('Directional Encoding')

plt.show()