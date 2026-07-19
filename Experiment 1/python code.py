import numpy as np
import matplotlib.pyplot as plt

x = np.array(input("Enter X values (comma-separated): ").split(","), dtype=float)
y = np.array(input("Enter Y values (comma-separated): ").split(","), dtype=float)

n = len(x)

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x * x)

slope = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_x2) - (sum_x ** 2))
intercept = (sum_y - slope * sum_x) / n

y_pred = slope * x + intercept

mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
plt.figure(figsize=(6, 4))
plt.scatter(x, y, color="green", label="Observed Values")
plt.plot(x, y_pred, color="blue", linewidth=2, label="Best Fit")
plt.title("Linear Regression")
plt.xlabel("Input Variable")
plt.ylabel("Predicted Output")
plt.legend()
plt.grid(True)
plt.show()

metrics = ["MSE", "MAE", "RMSE", "R² Score"]
values = [mse, mae, rmse, r2]

plt.figure(figsize=(6, 4))
plt.scatter(metrics, values, color="purple")

for i in range(len(metrics)):
    plt.text(metrics[i], values[i], f"{values[i]:.4f}", ha="center", va="bottom")

plt.title("Performance Evaluation")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Calculated Values")
plt.grid(True)
plt.show()

print("\n Linear Regression Result ")
print(f"Regression Equation : y = {slope:.4f}x + {intercept:.4f}")
print(" ")
print(f"Mean Squared Error (MSE)       : {mse:.4f}")
print(f"Mean Absolute Error (MAE)      : {mae:.4f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
print(f"R² Score   : {r2:.4f}")
print(" ")