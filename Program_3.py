import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data
target = iris.target
label_names = iris.target_names

iris_df = pd.DataFrame(data, columns=iris.feature_names)

pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
reduced_df['target'] = target

plt.figure(figsize=(10, 15))
colors = ['r', 'g', 'b']

for i, t in enumerate(np.unique(target)):
    subset = reduced_df[reduced_df['target'] == t]
    plt.scatter(subset['PC1'], subset['PC2'], 
                c=colors[i], 
                label=label_names[i])

plt.title('PCA on Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid()
plt.show()
