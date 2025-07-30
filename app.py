from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pandas as pd
from functions.operations import perform_operation
import json # Import json for chart data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max file size (e.g., 16MB)

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Store the uploaded DataFrame globally (consider a more robust solution for production)
df_storage = {}

@app.route('/')
def index():
    # Define available functions for display on the index page
    available_functions = [
        {"name": "Read File & Display", "func_name": "read_file"},
        {"name": "Get Row Labels", "func_name": "get_row_labels"},
        {"name": "Get Column Labels", "func_name": "get_column_labels"},
        {"name": "Return First N Rows", "func_name": "return_first_n_rows"},
        {"name": "Return Last N Rows", "func_name": "return_last_n_rows"},
        {"name": "Indexing & Selecting Data", "func_name": "indexing_selecting_data"},
        {"name": "Display Datatypes", "func_name": "display_datatypes"},
        {"name": "Count Unique Datatypes", "func_name": "count_unique_dtypes"},
        {"name": "Get Info", "func_name": "get_info"},
        {"name": "Detect Missing Values", "func_name": "detect_missing_values"},
        {"name": "Count Missing Values", "func_name": "count_missing_values"},
        {"name": "Frequency Distribution", "func_name": "frequency_distribution"},
        {"name": "Correlation", "func_name": "correlation"},
        {"name": "Statistical Summary", "func_name": "statistical_summary"},
        {"name": "Point & Range Estimates", "func_name": "point_range_estimates"},
        {"name": "Scatter Plot", "func_name": "scatter_plot"},
        {"name": "Histogram", "func_name": "histogram"},
        {"name": "Bar Plot", "func_name": "bar_plot"},
        {"name": "Whisker Plot (Box Plot)", "func_name": "box_plot"},
        {"name": "Pairwise Plot", "func_name": "pairwise_plot"},
        {"name": "Assign X and Y Axes", "func_name": "assign_axes"},
        {"name": "Mean", "func_name": "mean"},
        {"name": "Variance", "func_name": "variance"},
        {"name": "Standard Deviation", "func_name": "std_dev"},
        {"name": "Normal Distribution", "func_name": "normal_distribution"},
        {"name": "Left Tail Probability", "func_name": "left_tail_probability"},
        {"name": "Right Tail Probability", "func_name": "right_tail_probability"},
        {"name": "Probability Mass Function (PMF)", "func_name": "pmf"},
        {"name": "Probability Density Function (PDF)", "func_name": "pdf"},
        {"name": "Random Variable (RV)", "func_name": "random_variable"},
        {"name": "Cumulative Distribution Function (CDF)", "func_name": "cdf"},
        {"name": "Percent-point Function (PPF)", "func_name": "ppf"},
        {"name": "Generate Random Numbers", "func_name": "generate_random_numbers"},
        {"name": "Inverse CDF", "func_name": "inverse_cdf"},
        {"name": "Bounded Distribution", "func_name": "bounded_distribution"},
        {"name": "Binomial Distribution", "func_name": "binomial_distribution"},
        {"name": "Joint Probability", "func_name": "joint_probability"},
        {"name": "Marginal Probability", "func_name": "marginal_probability"},
        {"name": "Conditional Probability", "func_name": "conditional_probability"},
        {"name": "Perform Hypothesis Testing", "func_name": "hypothesis_testing"},
        {"name": "Get Rank of Matrix", "func_name": "get_rank_matrix"},
        {"name": "Get Number of Equations", "func_name": "get_number_equations"},
        {"name": "Perform Linear Algebra", "func_name": "linear_algebra"},
        {"name": "Gradient Descent Learning Rule", "func_name": "gradient_descent_learning_rule"},
        {"name": "Constrained Multivariable Optimization", "func_name": "constrained_optimization"},
        {"name": "Unconstrained Multivariable Optimization", "func_name": "unconstrained_optimization"},
        {"name": "Predictive Modelling", "func_name": "predictive_modelling"},
        {"name": "Linear Regression Analysis", "func_name": "linear_regression_analysis"},
        {"name": "Multiple Linear Regression", "func_name": "multiple_linear_regression"},
        {"name": "Decision Trees", "func_name": "decision_trees"},
        {"name": "Random Forests", "func_name": "random_forests"},
        {"name": "K-Nearest Neighbour", "func_name": "knn"},
        {"name": "K-Means Clustering", "func_name": "kmeans_clustering"},
        {"name": "Linear Discriminant Analysis (LDA)", "func_name": "lda"},
        {"name": "Logistic Regression", "func_name": "logistic_regression"},
        {"name": "Support Vector Machine (SVM)", "func_name": "svm"},
        {"name": "Model Evaluation", "func_name": "model_evaluation"},
        {"name": "Model Assumption", "func_name": "model_assumption"},
        {"name": "Principal Component Analysis (PCA)", "func_name": "pca"},
    ]
    return render_template('index.html', functions=available_functions)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        ext = file.filename.split('.')[-1].lower()
        df = None
        try:
            if ext == 'csv':
                df = pd.read_csv(filepath)
            elif ext in ['xls', 'xlsx']:
                df = pd.read_excel(filepath)
            elif ext == 'txt':
                # Attempt to read as CSV with tab delimiter for TXT, or adjust based on common TXT formats
                df = pd.read_csv(filepath, delimiter='\t')
            else:
                return jsonify({'error': 'Unsupported file type'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500

        df_storage['data'] = df
        preview_html = df.head().to_html(classes='table table-zebra w-full text-left border')
        # Return HTML for the preview, not just a success message
        return jsonify({'message': 'Upload successful', 'preview': preview_html})
    return jsonify({'error': 'No file received'}), 400

@app.route('/function/<func_name>')
def run_func(func_name):
    if 'data' not in df_storage:
        return redirect(url_for('index')) # Redirect if no file is uploaded

    df = df_storage['data']
    
    # Initialize chart data to empty for non-chart functions
    chart_labels = []
    chart_data = []
    chart_title = ''
    chart_type = '' # Default to empty, JS will handle
    
    try:
        result_html, chart_info = perform_operation(df, func_name)
        if chart_info:
            chart_labels = chart_info.get('labels', [])
            chart_data = chart_info.get('data', [])
            chart_title = chart_info.get('title', '')
            chart_type = chart_info.get('type', '')
    except Exception as e:
        result_html = f"<p class='text-red-500'>Error executing function: {str(e)}</p>"
        chart_info = {} # Ensure chart_info is defined even on error

    return render_template(
        "output.html",
        function=func_name, # Pass the function name for display
        result=result_html,
        chart_labels=json.dumps(chart_labels),  # Pass as JSON string
        chart_data=json.dumps(chart_data),      # Pass as JSON string
        chart_title=chart_title,
        chart_type=chart_type
    )

if __name__ == '__main__':
    app.run(debug=True)