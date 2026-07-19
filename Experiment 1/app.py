from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

result = {}

@app.route("/", methods=["GET", "POST"])
def home():
    global result

    if request.method == "POST":

        x = np.array(list(map(float, request.form["x"].split(","))))
        y = np.array(list(map(float, request.form["y"].split(","))))

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

        plt.figure(figsize=(6,4))
        plt.scatter(x, y, color="green", label="Observed Values")
        plt.plot(x, y_pred, color="blue", label="Best Fit")
        plt.title("Linear Regression")
        plt.xlabel("Input Variable")
        plt.ylabel("Predicted Output")
        plt.legend()
        plt.grid(True)
        plt.savefig("static/regression.png")
        plt.close()

        metrics = ["MSE","MAE","RMSE","R² Score"]
        values = [mse,mae,rmse,r2]

        plt.figure(figsize=(6,4))
        plt.scatter(metrics, values, color="purple")

        for i in range(4):
            plt.text(metrics[i], values[i], f"{values[i]:.4f}")

        plt.title("Performance Evaluation")
        plt.xlabel("Evaluation Metrics")
        plt.ylabel("Calculated Values")
        plt.grid(True)
        plt.savefig("static/performance.png")
        plt.close()

        result = {
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "mse": round(mse,4),
            "mae": round(mae,4),
            "rmse": round(rmse,4),
            "r2": round(r2,4)
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)