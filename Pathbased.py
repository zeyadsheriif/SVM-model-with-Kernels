import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

d5 = pd.read_csv("/content/Pathbased.txt", delimiter='\t', skiprows=7)
columns = ["x1", 'x2', 'y']
d5.columns = columns
x = d5[['x1', 'x2']]
y_train = d5['y']

scaler = StandardScaler()
scaler.fit(x)
X_train = scaler.transform(x)

svm = SVC(C=10000, kernel='rbf')
svm.fit(X_train, y_train)

accuracy = svm.score(X_train, y_train)
print("Accuracy:", accuracy*100,'%')

def plot_decision_regions(X, y, clf):
    unique_classes = np.unique(y)
    colors = ['red', 'blue', 'green']
    for cls, color in zip(unique_classes, colors):
        mask = (y == cls)
        plt.scatter(X[mask, 0], X[mask, 1], color=color, s=10)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    mesh_predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    mesh_predictions = mesh_predictions.reshape(xx.shape)

    for i, color in enumerate(colors):
        plt.scatter(xx[mesh_predictions == unique_classes[i]], yy[mesh_predictions == unique_classes[i]], marker='.', color=color, alpha=0.2, s=6)

    unclassified_points = np.where(~np.isin(mesh_predictions, unique_classes))
    plt.scatter(xx[unclassified_points], yy[unclassified_points], marker='.', color='white', alpha=0.2, s=6)

plot_decision_regions(X_train, y_train.values, clf=svm)
plt.grid(alpha=0.5)
plt.legend()
plt.show()
