import sys
sys.path.append('..')
from datasets import spiral
import matplotlib.pyplot as plt


x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
plt.show()