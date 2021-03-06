import matplotlib.pyplot as plt

w1 = [30.4, 5.51, 3.5, 2.67, 2.22, 1.92, 1.7, 1.53, 1.39, 1.28, 1.17, 1.08,
      0.99, 0.92, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.56, 0.51, 0.48, 0.4, 0.43, 0.36]
w2 = [9.61, 1.74, 1.11, 0.85, 0.7, 0.61, 0.54, 0.48, 0.44, 0.4, 0.37, 0.34, 0.31,
      0.29, 0.27, 0.25, 0.24, 0.22, 0.2, 0.19, 0.18, 0.16, 0.15, 0.13, 0.14, 0.11]
w3 = [6.83, 3.41, 2.51, 1.97, 1.6, 1.31, 1.06, 0.85, 0.66, 0.49, 0.31, 0.15, -0.01, -
      0.18, -0.32, -0.45, -0.59, -0.73, -0.87, -1.01, -1.18, -1.33, -1.48, -1.85, -1.69, -2.06]

plt.plot(w1, label='w1')
plt.plot(w2, label='w2')
plt.plot(w3, label='w3')
plt.legend()
plt.ylabel('Weight score')
plt.xlabel('x_i')

plt.savefig('weight_scores.png')
plt.show()
