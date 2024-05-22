import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def calculate_error_probability(n, k, epsilon):
    probabilities = [comb(n, i) * (epsilon ** i) * ((1 - epsilon) ** (n - i)) for i in range(k, n + 1)]
    return sum(probabilities)

epsilon = 0.45 
max_classifiers = 200 

n_values = np.arange(1, max_classifiers + 1, 2)
error_probabilities = [calculate_error_probability(n, n // 2 + 1, epsilon) for n in n_values]

threshold = 0.1
for n, prob in zip(n_values, error_probabilities):
    if prob < threshold:
        print(f"The first n where the error probability is less than {threshold} is {n}")
        break

plt.figure(figsize=(10, 6))
plt.plot(n_values, error_probabilities, marker='o')
plt.xlabel('Number of Weak Classifiers')
plt.ylabel('Error Probability')
plt.title('Error Probability vs Number of Weak Classifiers (Odd n)')
plt.grid(True)
plt.show()
