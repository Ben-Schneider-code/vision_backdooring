import matplotlib.pyplot as plt
import numpy as np


data = [.6, .955, .9968, .9988]
labels = ['75', '100', '125', '150']

#convert data to % from proportion
plt.bar(labels, np.array(data)*100)
plt.xlabel('Number of poisons [thousands]')
plt.ylabel('Acc (%)')
plt.title('Dendrogram Encoding')

plt.show()