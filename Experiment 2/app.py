import base64
import io
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

plt.style.use('dark_background')
DARK_BG = '#0f172a'
CARD_BG = '#1e293b'
ACCENT_GREEN = '#10b981'
ACCENT_BLUE = '#3b82f6'
TEXT_COLOR = '#f8fafc'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    y_name = request.form.get('y_name', 'y').strip()
    y_input = request.form.get('y_values', '')
    
    try:
        y_vals = np.array([float(val) for val in y_input.split(",") if val.strip()])
        num_data_points = len(y_vals)
    except ValueError:
        return "Error: Invalid numerical values for y.", 400

    x_data = {}
    x_names = []
    x_values_raw = []
    
    idx = 0
    while f'x_name_{idx}' in request.form:
        feat_name = request.form.get(f'x_name_{idx}').strip()
        feat_input = request.form.get(f'x_values_{idx}', '')
        
        if not feat_input.strip():
            idx += 1
            continue
            
        try:
            feat_vals = np.array([float(val) for val in feat_input.split(",") if val.strip()])
            if len(feat_vals) != num_data_points:
                return f"Error: '{feat_name}' has {len(feat_vals)} data points, but '{y_name}' has {num_data_points}.", 400
            
            x_names.append(feat_name)
            x_values_raw.append(feat_input)
            x_data[f'x{idx+1}'] = feat_vals
        except ValueError:
            return f"Error: Invalid numerical values inside feature '{feat_name}'.", 400
        idx += 1

    data = pd.DataFrame(x_data)
    model = LinearRegression()
    model.fit(data, y_vals)
    predictions = model.predict(data)

    mse = mean_squared_error(y_vals, predictions)
    mae = mean_absolute_error(y_vals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_vals, predictions)

    intercept = model.intercept_
    coefficients = model.coef_
    equation_terms = [f"({coef:.4f} * x{i+1})" for i, coef in enumerate(coefficients)]
    equation_str = f"y = {intercept:.4f} + " + " + ".join(equation_terms)
    slopes = {x_names[i]: f"{coef:.4f}" for i, coef in enumerate(coefficients)}

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=CARD_BG)
    ax.set_facecolor(DARK_BG)
    ax.scatter(y_vals, predictions, color=ACCENT_GREEN, edgecolor="#34d399", linewidth=1, s=60, alpha=0.9)
    min_val = min(float(np.min(y_vals)), float(np.min(predictions)))
    max_val = max(float(np.max(y_vals)), float(np.max(predictions)))
    ax.plot([min_val, max_val], [min_val, max_val], color="#94a3b8", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"Actual {y_name}", color=TEXT_COLOR)
    ax.set_ylabel(f"Predicted {y_name}", color=TEXT_COLOR)
    ax.set_title("Model Prediction Accuracy", color=TEXT_COLOR, pad=12, fontweight='bold')
    ax.grid(True, color="#334155", linestyle=":", alpha=0.6)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color('#334155')

    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format="png", facecolor=CARD_BG)
    buf1.seek(0)
    plot1_base64 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=CARD_BG)
    ax.set_facecolor(DARK_BG)
    metrics_names = ["R2 Score", "RMSE", "MAE", "MSE"]
    metrics_vals = [r2, rmse, mae, mse]
    colors = ["#6366f1", "#3b82f6", "#06b6d4", "#10b981"]
    bars = ax.barh(metrics_names, metrics_vals, color=colors, height=0.55)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f" {width:.4f}", xy=(width, bar.get_y() + bar.get_height() / 2),
                    ha='left', va='center', fontweight='bold', color=TEXT_COLOR, fontsize=9)
                    
    ax.set_xlabel("Computed Values", color=TEXT_COLOR)
    ax.set_title("Regression Metrics", color=TEXT_COLOR, pad=12, fontweight='bold')
    ax.grid(True, color="#334155", linestyle=":", alpha=0.6)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color('#334155')

    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format="png", facecolor=CARD_BG)
    buf2.seek(0)
    plot2_base64 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close()

    x_raw_str = ";".join(x_values_raw)

    return render_template('result.html', 
                           equation=equation_str, 
                           intercept=f"{intercept:.4f}", 
                           slopes=slopes, 
                           y_name=y_name,
                           y_input=y_input,
                           x_names=x_names,
                           x_raw_str=x_raw_str,
                           plot1=plot1_base64, 
                           plot2=plot2_base64)

@app.route('/predict', methods=['POST'])
def predict():
    y_input = request.form['y_input']
    x_raw_str = request.form['x_raw_str']
    y_name = request.form['y_name']

    y_vals = np.array([float(val) for val in y_input.split(",") if val.strip()])
    feature_strings = x_raw_str.split(";")
    
    x_data = {}
    for i, f_str in enumerate(feature_strings):
        x_data[f'x{i+1}'] = [float(val) for val in f_str.split(",") if val.strip()]
        
    model = LinearRegression()
    model.fit(pd.DataFrame(x_data), y_vals)

    try:
        sample_dict = {}
        for idx in range(len(feature_strings)):
            val = float(request.form[f'pred_x_{idx}'])
            sample_dict[f'x{idx+1}'] = val

        predicted_val = model.predict(pd.DataFrame([sample_dict]))[0]
        return f"""
        <html>
        <head>
            <style>
                body {{ background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
                .card {{ background: #1e293b; padding: 2.5rem; border-radius: 12px; border: 1px solid #334155; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }}
                .val {{ color: #10b981; font-size: 2.2rem; margin: 1rem 0; font-weight: 700; }}
                a {{ color: #3b82f6; text-decoration: none; font-weight: 600; display: inline-block; margin-top: 1rem; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h3 style="margin:0; color:#94a3b8;">Predicted {y_name} (y)</h3>
                <div class="val">{predicted_val:.4f}</div>
                <a href='javascript:history.back()'>← Back to Results</a>
            </div>
        </body>
        </html>
        """
    except Exception:
        return "Invalid input numbers entered for prediction.", 400

if __name__ == '__main__':
    app.run(debug=True)
