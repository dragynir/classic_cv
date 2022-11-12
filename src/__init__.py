import numpy as np
import matplotlib.pyplot as plt

image = np.random.randint(0, 255, (100, 100, 3))
threshold = 155

plt.imshow((image > threshold))




