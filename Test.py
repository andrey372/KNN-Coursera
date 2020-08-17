# Import libraries
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_digits
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import seaborn as sns


employee_numbers = [2, 9, 18, 28]
employee_names = ["Candice", "Ava", "Andrew", "Lucas"]


table = pd.DataFrame(employee_names, employee_numbers).reset_index()

print(table)