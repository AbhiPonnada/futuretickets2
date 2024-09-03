from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            
            # Read the Excel file
            data = pd.read_excel(filepath)
            
            # Process the file
            relevant_columns = ['Module', 'Created Time']
            data = data[relevant_columns]
            data['Created Time'] = pd.to_datetime(data['Created Time'])
            data['Month'] = data['Created Time'].dt.month
            data['Year'] = data['Created Time'].dt.year
            
            ticket_counts = data.groupby(['Module', 'Year', 'Month']).size().reset_index(name='Ticket Count')
            X = ticket_counts[['Year', 'Month', 'Module']]
            y = ticket_counts['Ticket Count']
            X = pd.get_dummies(X, columns=['Module'], drop_first=True)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            current_year = pd.Timestamp.now().year
            current_month = pd.Timestamp.now().month
            future_months = [(current_month + i - 1) % 12 + 1 for i in range(1, 13)]
            future_years = [current_year + (current_month + i - 1) // 12 for i in range(1, 13)]
            future_data = pd.DataFrame({
                'Year': future_years,
                'Month': future_months,
            })
            
            for col in X.columns[2:]:
                future_data[col] = 0
            future_data = future_data[X.columns]
            
            predicted_counts = []
            for module in X.columns[2:]:
                future_data_module = future_data.copy()
                future_data_module[module] = 1
                predicted_counts_module = model.predict(future_data_module)
                predicted_counts.append(predicted_counts_module)
            predicted_counts = np.array(predicted_counts).T
            
            plt.figure(figsize=(16, 10))
            bar_width = 0.2
            index = range(len(future_data))
            for i, module in enumerate(X.columns[2:]):
                plt.bar([x + i * bar_width for x in index], predicted_counts[:, i], width=bar_width, label=module)
            plt.xlabel('Month')
            plt.ylabel('Predicted Ticket Count')
            plt.title('Predicted Volume of Tickets by Module for Next 12 Months')
            plt.xticks(index, [f"{future_data['Month'][i]}-{future_data['Year'][i]}" for i in range(len(future_data))], rotation=45)
            plt.legend()
            plt.tight_layout()
            plot_filename = os.path.join("static", "plot.png")
            plt.savefig(plot_filename)
            plt.close()

            return render_template('index.html', plot_url=plot_filename)

    return render_template('index.html')
