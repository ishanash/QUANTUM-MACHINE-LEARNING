#!/usr/bin/env python
# coding: utf-8

# # QUANTUM MACHINE LEARNING (LINEAR REGRESSION)

# In[1]:


from qiskit import Aer, QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram,plot_bloch_multivector
import numpy as np


# In[2]:


def create_circuit(theta):
    qc = QuantumCircuit(1, 1)
    qc.ry(2 * theta, 0)
    qc.measure(0, 0)
    return qc


# In[3]:


def run_circuit(theta,shots):
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(create_circuit(theta), backend)
    qobj = assemble(transpiled_circuit)
    circuits = [transpiled_circuit]
    result = backend.run(circuits,shots=shots).result()
    counts = result.get_counts()
    return counts


# In[4]:


def calculate_accuracy(actual_theta, estimated_theta):
    estimated_theta = (estimated_theta + 2 * np.pi) % (2 * np.pi)
    error = abs(actual_theta - estimated_theta)
    accuracy = 1 - error / (2 * np.pi)
    return accuracy


# In[5]:


def visualize_estimation(results, theta):
    print(f"Chosen theta: {theta}")
    print(f"Estimation results: {results}")
    estimated_binary = list(results.keys())[0]
    estimated_theta = int(estimated_binary, 2) / (2 ** len(estimated_binary)) * 2 * np.pi
    print(f"Estimated theta: {estimated_theta}")
    accuracy = calculate_accuracy(theta, estimated_theta)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    plot_histogram(results, title=f"Estimation Results for Theta = {theta}")


# In[6]:


theta = np.pi / 4
circuit = create_circuit(theta)
shots_value=8192
results = run_circuit(theta,shots_value)
visualize_estimation(results,theta)


# In[7]:


print("Quantum Circuit:")
print(circuit)


# In[23]:


circuit.draw(output='mpl')


# In[9]:


print("Estimation Results:", results)


# In[10]:


plot_histogram(results)


# In[11]:


final_state = Aer.get_backend('statevector_simulator').run(create_circuit(theta)).result().get_statevector()
plot_bloch_multivector(final_state)


# # CLASSICAL LINEAR REGRESSION

# In[12]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt


# In[13]:


np.random.seed(42)
theta_true = np.pi / 4
X = np.random.rand(100, 1) * 4 * np.pi
y_true = np.sin(2 * X[:, 0])
noise = 0.1 * np.random.randn(100)
y_noisy = y_true + noise 


# In[14]:


model = LinearRegression()
model.fit(X, y_noisy)


# In[15]:


y_pred = model.predict(X)


# In[16]:


plt.scatter(X, y_noisy, label='Noisy Data')
plt.plot(X, y_true, label='True Function', color='red', linewidth=2)
plt.plot(X, y_pred, label='Linear Regression Fit', linestyle='--', color='green', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression for Parameter Estimation')
plt.show()


# In[17]:


mse = mean_squared_error(y_noisy, y_pred)
print(f'Mean Squared Error: {mse}')


# In[18]:


r2 = r2_score(y_noisy, y_pred)
print(f'R^2 Score: {r2}')


# In[20]:


tolerance = 0.1
accuracy_percentage = np.mean(np.abs(y_pred - y_noisy) < tolerance) * 100
print(f'Accuracy Percentage: {accuracy_percentage:.2f}%')


# In[ ]:




