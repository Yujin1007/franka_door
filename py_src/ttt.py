import matplotlib.pyplot as plt
import numpy as np

# Define the parameters
alpha = 0.1
time_steps = 300

# Initialize the value of 'a' for the first 100 and the next 100 time steps
a_values = np.concatenate((np.full(100, -10), np.full(100, 0),np.full(100, -10)))

# Initialize the filtered values list
filtered_values = [a_values[0]]

# Apply the low pass filter
for i in range(1, time_steps):
    filtered_value = alpha * a_values[i] + (1 - alpha) * filtered_values[-1]
    filtered_values.append(filtered_value)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(a_values, label='Original Value of a')
plt.plot(filtered_values, label='Filtered Value of a', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Low Pass Filtering of Value a')
plt.legend()
plt.grid(True)
plt.show()