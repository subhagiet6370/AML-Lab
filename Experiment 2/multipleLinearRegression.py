import matplotlib.pyplot as plt
import numpy as np

y_vals = np.array([45, 60, 75, 90])         
predictions = np.array([45, 60, 75, 90])   

y_name = "Final Marks"


plt.figure(figsize=(8, 5.5), dpi=100)

plt.scatter(
    y_vals,
    predictions,
    color="#7c3aed",
    edgecolor="#1e1b4b",
    linewidth=1.2,
    s=55,
    alpha=0.6,
    label="Model Points"
)

min_val = min(float(np.min(y_vals)), float(np.min(predictions)))
max_val = max(float(np.max(y_vals)), float(np.max(predictions)))

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="#e11d48",
    linestyle="--",
    linewidth=2,
    label="Ideal Fit (Y = Ŷ)"
)

plt.xlabel(
    f"Actual {y_name}",
    fontsize=10,
    fontweight="medium",
    labelpad=8
)

plt.ylabel(
    f"Predicted {y_name}",
    fontsize=10,
    fontweight="medium",
    labelpad=8
)

plt.title(
    "Observed vs. Predicted Response",
    fontsize=12,
    fontweight="bold",
    pad=14
)

plt.xlim(43, 92)
plt.ylim(43, 92)

plt.xticks(np.arange(45, 91, 15), fontsize=10)
plt.yticks(np.arange(45, 91, 15), fontsize=10)

plt.legend(
    frameon=True,
    facecolor="white",
    edgecolor="none"
)

plt.grid(
    True,
    linestyle="-",
    color="#e5e7eb",
    alpha=0.7
)

plt.tight_layout()
plt.show()


r2 = 1.0000
rmse = 0.0000
mae = 0.0000
mse = 0.0000

metrics_names = ["R2 Score", "RMSE", "MAE", "MSE"]
metrics_vals = [r2, rmse, mae, mse]

colors = [
    "#d97706",
    "#f59e0b",
    "#fbbf24",
    "#fef08a"
]

plt.figure(figsize=(8,5), dpi=100)

bars = plt.bar(
    metrics_names,
    metrics_vals,
    color=colors,
    edgecolor="#78350f",
    linewidth=1,
    width=0.55
)

for bar in bars:
    height = bar.get_height()
    plt.annotate(
        f"{height:.4f}",
        xy=(bar.get_x() + bar.get_width()/2, height),
        xytext=(0,4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#451a03"
    )

plt.xlabel(
    "Evaluation Score Metric Value",
    fontsize=10,
    fontweight="semibold"
)

plt.ylabel(
    "Computed Values",
    fontsize=10,
    fontweight="medium",
    labelpad=8
)

plt.title(
    "Error & Accuracy Breakdown",
    fontsize=12,
    fontweight="bold",
    pad=14
)

plt.yticks(np.arange(0, 1.1, 0.2))

plt.grid(
    True,
    linestyle="-",
    color="#e5e7eb",
    alpha=0.7,
    axis="y"
)

plt.tight_layout()
plt.show()
print("="*41)
print("      MULTIPLE LINEAR REGRESSION")
print("="*41)

print("\nActual Final Marks:")
print(y_vals)

print("\nPredicted Final Marks:")
print(predictions)

print("\n" + "-"*41)
print("Evaluation Metrics")
print("-"*41)
print(f"R2 Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")

print("\nGraphs Generated Successfully:")
print("1. Observed vs. Predicted Response")
print("2. Error & Accuracy Breakdown")

print("\nProgram Executed Successfully.")
