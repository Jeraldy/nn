import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("ex2data1.txt")

def visualize_data():
    x1 = dataset.loc[:,['Test 1']].as_matrix()
    x2 = dataset.loc[:,['Test 2']].as_matrix()
    y = dataset.loc[:,["Admission status"]].as_matrix()
    plt.scatter(x1, x2, s=40, c=y, cmap=plt.cm.Spectral)
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    return plt.show()

if __name__ == '__main__':
    visualize_data()