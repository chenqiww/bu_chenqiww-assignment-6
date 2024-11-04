from flask import Flask, render_template, request, flash
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    hist_url = None
    more_extreme_slope = None
    more_extreme_intercept = None
    if request.method == 'POST':
        try:
            # Get form inputs
            N = int(request.form['N'])
            mu = float(request.form['mu'])
            sigma2 = float(request.form['sigma2'])
            S = int(request.form['S'])

            # Input validation
            N_max = 10000
            S_max = 10000
            error_message = None
            if N <= 0 or N > N_max:
                error_message = f"Sample size N must be a positive integer less than or equal to {N_max}."
            elif S <= 0 or S > S_max:
                error_message = f"Number of simulations S must be a positive integer less than or equal to {S_max}."
            elif sigma2 < 0:
                error_message = "Variance sigma² must be non-negative."

            if error_message:
                flash(error_message)
                return render_template('index.html', N=N, mu=mu, sigma2=sigma2, S=S)
            
            # Generate initial dataset
            X = np.random.uniform(0, 1, N)
            epsilon = np.random.normal(mu, np.sqrt(sigma2), N)
            Y = epsilon  # No relationship between X and Y
            
            # Fit linear regression to initial dataset
            model = LinearRegression()
            X_reshaped = X.reshape(-1, 1)
            model.fit(X_reshaped, Y)
            intercept = model.intercept_
            slope = model.coef_[0]
            
            # Generate scatter plot with regression line
            fig, ax = plt.subplots()
            ax.scatter(X, Y, label='Data Points')
            # Sort X for a smoother line
            X_sorted = np.sort(X)
            X_sorted_reshaped = X_sorted.reshape(-1, 1)
            Y_pred_sorted = model.predict(X_sorted_reshaped)
            ax.plot(X_sorted, Y_pred_sorted, color='red', label='Fitted Line')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Linear Fit: Y = {intercept:.2f} + {slope:.2f}X')
            ax.legend()
            figfile = io.BytesIO()
            plt.savefig(figfile, format='png')
            plt.close(fig)
            figfile.seek(0)
            plot_url = base64.b64encode(figfile.getvalue()).decode('utf8')
            
            # Simulate S datasets and compute slopes and intercepts
            slopes = []
            intercepts = []
            for _ in range(S):
                X_sim = np.random.uniform(0, 1, N)
                epsilon_sim = np.random.normal(mu, np.sqrt(sigma2), N)
                Y_sim = epsilon_sim
                model_sim = LinearRegression()
                model_sim.fit(X_sim.reshape(-1, 1), Y_sim)
                slopes.append(model_sim.coef_[0])
                intercepts.append(model_sim.intercept_)
            
            # Generate histograms
            fig2, ax2 = plt.subplots()
            bins = np.linspace(min(slopes + intercepts), max(slopes + intercepts), 30)
            ax2.hist(slopes, bins=bins, alpha=0.5, label='Slopes', color='blue')
            ax2.hist(intercepts, bins=bins, alpha=0.5, label='Intercepts', color='orange')
            ax2.axvline(slope, color='blue', linestyle='solid', linewidth=2, label=f'Calculated Slope: {slope:.2f}')
            ax2.axvline(intercept, color='orange', linestyle='dashed', linewidth=2, label=f'Calculated Intercept: {intercept:.2f}')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Histogram of Slopes and Intercepts')
            ax2.legend()
            figfile2 = io.BytesIO()
            plt.savefig(figfile2, format='png')
            plt.close(fig2)
            figfile2.seek(0)
            hist_url = base64.b64encode(figfile2.getvalue()).decode('utf8')
            
            # Compute proportions of more extreme slopes and intercepts
            more_extreme_slope = np.mean(np.abs(slopes) >= np.abs(slope))
            more_extreme_intercept = np.mean(np.abs(intercepts) >= np.abs(intercept))
            
            # Pass the input values back to the template
            return render_template('index.html', N=N, mu=mu, sigma2=sigma2, S=S, plot_url=plot_url, hist_url=hist_url, more_extreme_slope=more_extreme_slope, more_extreme_intercept=more_extreme_intercept)
        except ValueError:
            # Handle the case where inputs are not numbers
            error_message = "Please enter valid numeric values for all fields."
            flash(error_message)
            # Re-populate fields with the last inputs if available
            N = request.form.get('N', 100)
            mu = request.form.get('mu', 0)
            sigma2 = request.form.get('sigma2', 1)
            S = request.form.get('S', 1000)
            return render_template('index.html', N=N, mu=mu, sigma2=sigma2, S=S)
    else:
        # Set default values
        N = 100
        mu = 0
        sigma2 = 1
        S = 1000
        return render_template('index.html', N=N, mu=mu, sigma2=sigma2, S=S)
    
if __name__ == '__main__':
    app.run(debug=True)
