import matplotlib.pyplot as plt
import numpy as np


data = [.001, .011, .14, .82]
labels = ['10', '50', '75', '100']

#convert data to % from proportion
plt.bar(labels, np.array(data)*100)
plt.xlabel('Number of poisons [thousands]')
plt.ylabel('Acc (%)')
plt.title('Naive Badnet Enumeration @ 5 epochs [unshuffled]')

plt.show()