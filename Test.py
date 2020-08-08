# Import libraries
from sklearn import datasets
from matplotlib import pyplot as plt

# Get regression data from scikit-learn
x, y = datasets.make_regression(n_samples=20, n_features=1, noise=0.5)

# Vizualize the data
plt.scatter(x, y)
plt.show()