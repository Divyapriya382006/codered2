import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, base64
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from scipy.stats import norm, binom # For probability distributions
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import statsmodels.api as sm # For OLS for point/range estimates, hypothesis testing

def render_plot(fig):
    """Encodes a matplotlib figure to a base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') # bbox_inches for better plot saving
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # Close the figure to free up memory
    return f'<img src="data:image/png;base64,{img}" class="rounded-xl shadow-lg mx-auto" />'

def perform_operation(df, operation):
    """
    Performs the specified data operation on the DataFrame.
    Returns HTML string for display and optionally chart data.
    """
    result_html = ""
    chart_info = {} # To store labels, data, title, type for charts

    try:
        # FILES accepted: txt, csv,excel
        # Read file and display file (This is handled by the upload logic and preview)
        if operation == "read_file":
            result_html = df.to_html(classes="table-auto w-full", max_rows=100)
            return result_html, chart_info

        # Get row labels
        elif operation == "get_row_labels":
            result_html = f"<p>Row Labels: {list(df.index)}</p>"
            return result_html, chart_info

        # Get column labels
        elif operation == "get_column_labels":
            result_html = f"<p>Column Labels: {list(df.columns)}</p>"
            return result_html, chart_info

        # Return first n rows
        elif operation == "return_first_n_rows":
            # For simplicity, we'll return the first 5 rows. In a real app, 'n' would be an input.
            result_html = df.head(5).to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Return last n rows
        elif operation == "return_last_n_rows":
            # For simplicity, we'll return the last 5 rows.
            result_html = df.tail(5).to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Indexing and selecting data (Example: first 5 rows, first 2 columns)
        elif operation == "indexing_selecting_data":
            result_html = df.iloc[:5, :2].to_html(classes="table-auto w-full")
            result_html += "<p><em>Displayed first 5 rows and first 2 columns as an example.</em></p>"
            return result_html, chart_info

        # Display datatypes
        elif operation == "display_datatypes":
            result_html = df.dtypes.to_frame("Data Type").to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Get counts of unique data types
        elif operation == "count_unique_dtypes":
            dtype_counts = df.dtypes.value_counts().to_frame("Count of Unique Data Types")
            result_html = dtype_counts.to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Get info
        elif operation == "get_info":
            buf = io.StringIO()
            df.info(buf=buf)
            result_html = f"<pre class='bg-gray-700 p-4 rounded-md text-white overflow-x-auto'>{buf.getvalue()}</pre>"
            return result_html, chart_info

        # Detect missing values
        elif operation == "detect_missing_values":
            missing_data = df.isnull().any().to_frame("Has Missing Values")
            result_html = missing_data.to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Count number of missing values
        elif operation == "count_missing_values":
            missing_counts = df.isnull().sum().to_frame("Number of Missing Values")
            result_html = missing_counts.to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Frequency distribution (for the first categorical column)
        elif operation == "frequency_distribution":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                freq_dist = df[categorical_cols[0]].value_counts().to_frame("Frequency")
                result_html = freq_dist.to_html(classes="table-auto w-full")
                result_html += f"<p><em>Showing frequency distribution for '{categorical_cols[0]}'.</em></p>"
                
                # Chart data for bar plot
                chart_info = {
                    'labels': freq_dist.index.tolist(),
                    'data': freq_dist['Frequency'].tolist(),
                    'title': f'Frequency Distribution of {categorical_cols[0]}',
                    'type': 'bar'
                }
            else:
                result_html = "<p class='text-yellow-500'>No categorical columns found for frequency distribution.</p>"
            return result_html, chart_info

        # Correlation
        elif operation == "correlation":
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                correlation_matrix = numeric_df.corr()
                result_html = correlation_matrix.to_html(classes="table-auto w-full")
                
                # Heatmap for correlation
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Correlation Matrix')
                result_html += render_plot(fig)
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for correlation calculation.</p>"
            return result_html, chart_info

        # Statistical summary of data
        elif operation == "statistical_summary":
            result_html = df.describe().to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Point and range estimates (e.g., mean and confidence interval for the first numeric column)
        elif operation == "point_range_estimates":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data_col = df[numeric_cols[0]].dropna()
                mean_val = data_col.mean()
                std_err = data_col.sem() # Standard error of the mean
                
                # 95% Confidence Interval for the mean
                lower_bound, upper_bound = norm.interval(0.95, loc=mean_val, scale=std_err)
                
                result_html = f"<p>Point Estimate (Mean) for '{numeric_cols[0]}': {mean_val:.4f}</p>"
                result_html += f"<p>95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for point and range estimates.</p>"
            return result_html, chart_info

        # 📊 Visualization (6):
        # Scatter plot (first two numeric columns)
        elif operation == "scatter_plot":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], ax=ax)
                ax.set_title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                result_html = render_plot(fig)
            else:
                result_html = "<p class='text-yellow-500'>Need at least two numeric columns for a scatter plot.</p>"
            return result_html, chart_info

        # Histogram (first numeric column)
        elif operation == "histogram":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                fig, ax = plt.subplots()
                sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=ax)
                ax.set_title(f'Histogram of {numeric_cols[0]}')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel('Frequency')
                result_html = render_plot(fig)
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for a histogram.</p>"
            return result_html, chart_info

        # Bar plot (first categorical column, value counts)
        elif operation == "bar_plot":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                counts = df[categorical_cols[0]].value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Bar Plot of {categorical_cols[0]}')
                ax.set_xlabel(categorical_cols[0])
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                result_html = render_plot(fig)
            else:
                result_html = "<p class='text-yellow-500'>No categorical columns found for a bar plot.</p>"
            return result_html, chart_info

        # Whisker plot (Box plot) (all numeric columns)
        elif operation == "box_plot":
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=numeric_df, ax=ax)
                ax.set_title('Box Plot of Numeric Columns')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                result_html = render_plot(fig)
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for a box plot.</p>"
            return result_html, chart_info

        # Pairwise plot (numeric columns, requires seaborn)
        elif operation == "pairwise_plot":
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) >= 2: # At least two numeric columns for a meaningful pairplot
                # For large datasets, pairplot can be slow and memory intensive. Limit to a few columns.
                subset_df = numeric_df.iloc[:, :min(numeric_df.shape[1], 5)] # Limit to max 5 columns
                if subset_df.shape[1] > 1:
                    fig = sns.pairplot(subset_df)
                    fig.fig.suptitle('Pairwise Plot of Numeric Columns (Subset)', y=1.02) # Add title to the figure
                    result_html = render_plot(fig.fig) # Pass the figure object
                else:
                    result_html = "<p class='text-yellow-500'>Not enough numeric columns (at least two) for a pairwise plot in the subset.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for a pairwise plot.</p>"
            return result_html, chart_info

        # Assign x and y axes (this function would typically involve user input for column selection,
        # but for demonstration, we'll just display a message)
        elif operation == "assign_axes":
            result_html = "<p>This function would allow you to select X and Y axes for plotting. Please implement user input for this.</p>"
            return result_html, chart_info

        # 📐 Descriptive Stats (3):
        # Mean
        elif operation == "mean":
            result_html = df.mean(numeric_only=True).to_frame("Mean").to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Variance
        elif operation == "variance":
            result_html = df.var(numeric_only=True).to_frame("Variance").to_html(classes="table-auto w-full")
            return result_html, chart_info

        # Standard deviation
        elif operation == "std_dev":
            result_html = df.std(numeric_only=True).to_frame("Standard Deviation").to_html(classes="table-auto w-full")
            return result_html, chart_info

        # 🧠 Probability & Distributions (18):
        # Normal distributions (PDF for the first numeric column)
        elif operation == "normal_distribution":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    mu, std = data.mean(), data.std()
                    x = np.linspace(data.min(), data.max(), 100)
                    pdf = norm.pdf(x, mu, std)
                    
                    fig, ax = plt.subplots()
                    sns.histplot(data, bins=30, kde=False, stat='density', alpha=0.6, label='Data Histogram', ax=ax)
                    ax.plot(x, pdf, 'r-', lw=2, label='Normal PDF')
                    ax.set_title(f'Normal Distribution (PDF) for {numeric_cols[0]}')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel('Probability Density')
                    ax.legend()
                    result_html = render_plot(fig)
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for normal distribution.</p>"
            return result_html, chart_info

        # Left tail probability (Example: P(X < mean - std) for first numeric column)
        elif operation == "left_tail_probability":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    mu, std = data.mean(), data.std()
                    # Example threshold: one standard deviation below the mean
                    threshold = mu - std
                    prob = norm.cdf(threshold, loc=mu, scale=std)
                    result_html = f"<p>Left Tail Probability for '{numeric_cols[0]}' (X < {threshold:.4f}): {prob:.4f}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for left tail probability.</p>"
            return result_html, chart_info

        # Right tail probability (Example: P(X > mean + std) for first numeric column)
        elif operation == "right_tail_probability":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    mu, std = data.mean(), data.std()
                    # Example threshold: one standard deviation above the mean
                    threshold = mu + std
                    prob = 1 - norm.cdf(threshold, loc=mu, scale=std)
                    result_html = f"<p>Right Tail Probability for '{numeric_cols[0]}' (X > {threshold:.4f}): {prob:.4f}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for right tail probability.</p>"
            return result_html, chart_info

        # Probability mass function (PMF) (for first integer column assuming discrete data)
        elif operation == "pmf":
            integer_cols = df.select_dtypes(include=['int64', 'int32']).columns
            if not integer_cols.empty:
                data = df[integer_cols[0]].dropna()
                if not data.empty:
                    # For demonstration, let's consider a simple count of unique values as PMF
                    pmf_series = data.value_counts(normalize=True).sort_index()
                    result_html = pmf_series.to_frame("PMF").to_html(classes="table-auto w-full")
                    
                    fig, ax = plt.subplots()
                    pmf_series.plot(kind='bar', ax=ax)
                    ax.set_title(f'Probability Mass Function for {integer_cols[0]}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Probability')
                    plt.tight_layout()
                    result_html += render_plot(fig)
                else:
                    result_html = "<p class='text-yellow-500'>Selected integer column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No integer columns found for PMF.</p>"
            return result_html, chart_info

        # Probability density function (PDF) (for first numeric column, same as normal_distribution for continuous)
        elif operation == "pdf":
            # This is essentially the same as normal_distribution for continuous data
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    mu, std = data.mean(), data.std()
                    x = np.linspace(data.min(), data.max(), 100)
                    pdf = norm.pdf(x, mu, std)
                    
                    fig, ax = plt.subplots()
                    sns.histplot(data, bins=30, kde=False, stat='density', alpha=0.6, label='Data Histogram', ax=ax)
                    ax.plot(x, pdf, 'r-', lw=2, label='Estimated PDF')
                    ax.set_title(f'Probability Density Function for {numeric_cols[0]}')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel('Probability Density')
                    ax.legend()
                    result_html = render_plot(fig)
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for PDF.</p>"
            return result_html, chart_info

        # Random variable (RV) (sample 5 random values from the first numeric column)
        elif operation == "random_variable":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    # Ensure there are enough samples to draw
                    num_samples = min(5, len(data))
                    random_samples = data.sample(num_samples).tolist()
                    result_html = f"<p>Random Samples from '{numeric_cols[0]}': {random_samples}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for random variable sampling.</p>"
            return result_html, chart_info

        # Cumulative distribution function (CDF) (for first numeric column)
        elif operation == "cdf":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    sorted_data = np.sort(data)
                    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    fig, ax = plt.subplots()
                    ax.plot(sorted_data, cdf, marker='.', linestyle='none', markersize=4)
                    ax.set_title(f'Cumulative Distribution Function (CDF) for {numeric_cols[0]}')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel('CDF')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    result_html = render_plot(fig)
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for CDF.</p>"
            return result_html, chart_info

        # Percent-point function (PPF) (Inverse CDF, e.g., 25th, 50th, 75th percentiles for first numeric column)
        elif operation == "ppf":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    percentiles = [25, 50, 75]
                    ppf_values = np.percentile(data, percentiles)
                    
                    result_html = f"<p>Percent-Point Function (PPF) values for '{numeric_cols[0]}':</p>"
                    for p, val in zip(percentiles, ppf_values):
                        result_html += f"<p>{p}th Percentile: {val:.4f}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for PPF.</p>"
            return result_html, chart_info

        # Generate random numbers (example: 10 random numbers from a normal distribution based on dataset stats)
        elif operation == "generate_random_numbers":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    mu, std = data.mean(), data.std()
                    random_numbers = norm.rvs(loc=mu, scale=std, size=10)
                    result_html = f"<p>10 Random Numbers from a Normal Distribution (mean={mu:.2f}, std={std:.2f}): {random_numbers.tolist()}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found to base random number generation on.</p>"
            return result_html, chart_info

        # Inverse cumulative distribution function (same as PPF)
        elif operation == "inverse_cdf":
            # This is essentially the same as PPF
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    percentiles = [25, 50, 75]
                    ppf_values = np.percentile(data, percentiles)
                    
                    result_html = f"<p>Inverse CDF (PPF) values for '{numeric_cols[0]}':</p>"
                    for p, val in zip(percentiles, ppf_values):
                        result_html += f"<p>{p}th Percentile: {val:.4f}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for Inverse CDF.</p>"
            return result_html, chart_info

        # Bounded distribution (example: Uniform distribution, specify min/max based on data)
        elif operation == "bounded_distribution":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    min_val, max_val = data.min(), data.max()
                    # Example: generate 10 random numbers from a uniform distribution within data bounds
                    random_uniform = np.random.uniform(low=min_val, high=max_val, size=10)
                    result_html = f"<p>10 Random Numbers from a Uniform Distribution [{min_val:.2f}, {max_val:.2f}]: {random_uniform.tolist()}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found to define bounds.</p>"
            return result_html, chart_info

        # Binomial distribution (Example: using a success probability derived from a binary column)
        elif operation == "binomial_distribution":
            # Assuming a binary column (0/1) or one that can be converted to such
            binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].isin([0, 1]).all()]
            if not binary_cols:
                # Try to find a boolean column
                binary_cols = [col for col in df.columns if df[col].dtype == 'bool']

            if binary_cols:
                data_col = df[binary_cols[0]].dropna()
                if not data_col.empty:
                    n_trials = len(data_col) # Number of trials (data points)
                    p_success = data_col.mean() # Probability of success (mean of binary column)
                    
                    # Calculate PMF for a few possible number of successes (k)
                    k_values = range(int(n_trials / 2) - 2, int(n_trials / 2) + 3) # Example k values around mean
                    pmf_values = [binom.pmf(k, n_trials, p_success) for k in k_values]
                    
                    result_html = f"<p>Binomial Distribution (n={n_trials}, p={p_success:.4f}) for '{binary_cols[0]}':</p>"
                    for k, pmf in zip(k_values, pmf_values):
                        result_html += f"<p>P(X={k}): {pmf:.4f}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected binary column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No suitable binary column found for Binomial Distribution (requires 0/1 or boolean column).</p>"
            return result_html, chart_info

        # Joint probability (from first two categorical columns)
        elif operation == "joint_probability":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) >= 2:
                joint_prob_table = pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]], normalize=True)
                result_html = "<p>Joint Probability Table:</p>"
                result_html += joint_prob_table.to_html(classes="table-auto w-full")
            else:
                result_html = "<p class='text-yellow-500'>Need at least two categorical columns for joint probability.</p>"
            return result_html, chart_info

        # Marginal probability (from first categorical column)
        elif operation == "marginal_probability":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                marginal_prob_series = df[categorical_cols[0]].value_counts(normalize=True)
                result_html = "<p>Marginal Probability for '{}':</p>".format(categorical_cols[0])
                result_html += marginal_prob_series.to_frame("Marginal Probability").to_html(classes="table-auto w-full")
            else:
                result_html = "<p class='text-yellow-500'>No categorical columns found for marginal probability.</p>"
            return result_html, chart_info

        # Conditional probability (Example: P(Col2 | Col1) for first two categorical columns)
        elif operation == "conditional_probability":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) >= 2:
                # P(Col2 | Col1) = P(Col1 and Col2) / P(Col1)
                conditional_prob_table = pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]], normalize='index')
                result_html = "<p>Conditional Probability P({} | {}):</p>".format(categorical_cols[1], categorical_cols[0])
                result_html += conditional_prob_table.to_html(classes="table-auto w-full")
            else:
                result_html = "<p class='text-yellow-500'>Need at least two categorical columns for conditional probability.</p>"
            return result_html, chart_info

        # 🧪 Hypothesis Testing (1):
        # Perform hypothesis testing (e.g., t-test for mean of first numeric column against 0)
        elif operation == "hypothesis_testing":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                data = df[numeric_cols[0]].dropna()
                if not data.empty:
                    # Example: One-sample t-test (H0: mean = 0)
                    from scipy import stats
                    t_statistic, p_value = stats.ttest_1samp(data, 0)
                    
                    alpha = 0.05
                    conclusion = f"Fail to reject H0 (p > {alpha})" if p_value > alpha else f"Reject H0 (p <= {alpha})"
                    
                    result_html = f"<p>One-Sample T-test for '{numeric_cols[0]}' (H0: mean = 0):</p>"
                    result_html += f"<p>T-statistic: {t_statistic:.4f}</p>"
                    result_html += f"<p>P-value: {p_value:.4f}</p>"
                    result_html += f"<p>Conclusion at $\\alpha$={alpha}: {conclusion}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>Selected numeric column is empty after dropping NaNs.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found for hypothesis testing.</p>"
            return result_html, chart_info

        # 🧮 Linear Algebra (4):
        # Get rank of matrix (from first few numeric columns)
        elif operation == "get_rank_matrix":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] >= 2: # Need at least 2 columns to form a matrix for rank
                matrix = numeric_df.to_numpy()
                rank = np.linalg.matrix_rank(matrix)
                result_html = f"<p>Rank of the numeric data matrix: {rank}</p>"
                result_html += "<p><em>Calculated from all numeric columns.</em></p>"
            else:
                result_html = "<p class='text-yellow-500'>Need at least two numeric columns to form a matrix for rank calculation.</p>"
            return result_html, chart_info

        # Get number of equations (if data represents a system of linear equations, could be number of rows)
        elif operation == "get_number_equations":
            result_html = f"<p>Number of rows (could represent number of equations): {df.shape[0]}</p>"
            return result_html, chart_info

        # Perform linear algebra (example: matrix multiplication of first few numeric columns with its transpose)
        elif operation == "linear_algebra":
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                # Ensure the matrix is not too large for basic operations
                matrix_a = numeric_df.iloc[:, :min(numeric_df.shape[1], 3)].to_numpy() # Limit to 3 columns
                if matrix_a.shape[1] > 0:
                    # Example: A * A_transpose
                    try:
                        result_matrix = np.dot(matrix_a.T, matrix_a)
                        result_html = "<p>Result of A_transpose * A (from first few numeric columns):</p>"
                        result_html += pd.DataFrame(result_matrix).to_html(classes="table-auto w-full")
                    except Exception as e:
                        result_html = f"<p class='text-red-500'>Error performing matrix multiplication: {e}</p>"
                else:
                    result_html = "<p class='text-yellow-500'>No numeric columns to perform linear algebra.</p>"
            else:
                result_html = "<p class='text-yellow-500'>No numeric columns found to perform linear algebra.</p>"
            return result_html, chart_info

        # Gradient descent learning rule (Conceptual, often part of ML model training, not a standalone output)
        elif operation == "gradient_descent_learning_rule":
            result_html = "<p>Gradient Descent is an optimization algorithm often used in machine learning to minimize a cost function. Its 'rule' involves iteratively adjusting parameters in the direction opposite to the gradient of the cost function.</p>"
            result_html += "<p><em>This is a conceptual explanation; a practical implementation requires a specific model and objective function.</em></p>"
            return result_html, chart_info

        # 🧠 Optimization (2):
        # Constrained multivariable optimization (Conceptual, requires defining constraints and objective)
        elif operation == "constrained_optimization":
            result_html = "<p>Constrained multivariable optimization involves finding the optimal values of multiple variables subject to certain constraints (e.g., resource limits, non-negativity).</p>"
            result_html += "<p><em>This requires a specific problem definition with objective function and constraints.</em></p>"
            return result_html, chart_info

        # Unconstrained multivariable optimization (Conceptual, requires defining objective function)
        elif operation == "unconstrained_optimization":
            result_html = "<p>Unconstrained multivariable optimization involves finding the optimal values of multiple variables without any explicit constraints. Methods often involve finding where the gradient of the objective function is zero.</p>"
            result_html += "<p><em>This requires a specific objective function.</em></p>"
            return result_html, chart_info

        # 🤖 Machine Learning Models (15):
        # Predictive modelling (General term, covered by specific models below)
        elif operation == "predictive_modelling":
            result_html = "<p>Predictive modelling involves building models to forecast future outcomes or identify patterns. Specific models like Linear Regression, Decision Trees, etc., are examples.</p>"
            return result_html, chart_info

        # Linear regression analysis (Simple linear regression with first two numeric columns)
        elif operation == "linear_regression_analysis":
            numeric_df = df.select_dtypes(include=np.number).dropna()
            if len(numeric_df.columns) >= 2:
                X = numeric_df.iloc[:, 0].values.reshape(-1, 1) # Independent variable
                y = numeric_df.iloc[:, 1].values # Dependent variable
                
                if X.size > 0 and y.size > 0:
                    model = LinearRegression()
                    model.fit(X, y)
                    r_squared = model.score(X, y)
                    
                    result_html = f"<p>Simple Linear Regression ({numeric_df.columns[0]} vs {numeric_df.columns[1]}):</p>"
                    result_html += f"<p>Coefficient: {model.coef_[0]:.4f}</p>"
                    result_html += f"<p>Intercept: {model.intercept_:.4f}</p>"
                    result_html += f"<p>R-squared: {r_squared:.4f}</p>"
                    
                    # Plotting the regression line
                    fig, ax = plt.subplots()
                    ax.scatter(X, y, color='blue', label='Actual Data')
                    ax.plot(X, model.predict(X), color='red', label='Regression Line')
                    ax.set_title(f'Linear Regression: {numeric_df.columns[0]} vs {numeric_df.columns[1]}')
                    ax.set_xlabel(numeric_df.columns[0])
                    ax.set_ylabel(numeric_df.columns[1])
                    ax.legend()
                    result_html += render_plot(fig)
                else:
                    result_html = "<p class='text-yellow-500'>Not enough valid data in selected numeric columns for linear regression.</p>"
            else:
                result_html = "<p class='text-yellow-500'>Need at least two numeric columns for Linear Regression Analysis.</p>"
            return result_html, chart_info

        # Multiple linear regression (Requires at least 3 numeric columns: multiple features and one target)
        elif operation == "multiple_linear_regression":
            numeric_df = df.select_dtypes(include=np.number).dropna()
            if numeric_df.shape[1] >= 3:
                # X: First two numeric columns, y: Third numeric column
                X = numeric_df.iloc[:, :2]
                y = numeric_df.iloc[:, 2]
                
                # Add a constant for the intercept term for statsmodels OLS
                X = sm.add_constant(X) 
                
                model = sm.OLS(y, X).fit()
                result_html = "<p>Multiple Linear Regression Summary:</p>"
                result_html += model.summary().as_html()
            else:
                result_html = "<p class='text-yellow-500'>Need at least three numeric columns for Multiple Linear Regression (two features, one target).</p>"
            return result_html, chart_info

        # Decision trees (Classification example, requires a target variable)
        elif operation == "decision_trees":
            # Assuming the last column is the target, encode categorical if any
            temp_df = df.copy()
            for col in temp_df.select_dtypes(include='object').columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col])
            
            if temp_df.shape[1] < 2:
                result_html = "<p class='text-yellow-500'>Not enough columns for Decision Tree classification (need at least one feature and a target).</p>"
                return result_html, chart_info

            X = temp_df.iloc[:, :-1].dropna()
            y = temp_df.iloc[:, -1].dropna()

            if X.empty or y.empty or len(y.unique()) < 2:
                result_html = "<p class='text-yellow-500'>Target variable not suitable for classification or insufficient data after dropping NaNs.</p>"
                return result_html, chart_info
            
            # Ensure X and y have the same number of rows after dropping NaNs
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.empty or y.empty or len(y.unique()) < 2:
                 result_html = "<p class='text-yellow-500'>Insufficient data or target variable issues after alignment and NaN handling.</p>"
                 return result_html, chart_info


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            
            result_html = f"<p>Decision Tree Classifier Accuracy: {accuracy:.4f}</p>"
            result_html += f"<p>Confusion Matrix:<pre>{confusion_matrix(y_test, preds)}</pre></p>"
            return result_html, chart_info

        # Random forests (Classification example, requires a target variable)
        elif operation == "random_forests":
            temp_df = df.copy()
            for col in temp_df.select_dtypes(include='object').columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col])

            if temp_df.shape[1] < 2:
                result_html = "<p class='text-yellow-500'>Not enough columns for Random Forest classification (need at least one feature and a target).</p>"
                return result_html, chart_info

            X = temp_df.iloc[:, :-1].dropna()
            y = temp_df.iloc[:, -1].dropna()

            if X.empty or y.empty or len(y.unique()) < 2:
                result_html = "<p class='text-yellow-500'>Target variable not suitable for classification or insufficient data after dropping NaNs.</p>"
                return result_html, chart_info

            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.empty or y.empty or len(y.unique()) < 2:
                 result_html = "<p class='text-yellow-500'>Insufficient data or target variable issues after alignment and NaN handling.</p>"
                 return result_html, chart_info
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            
            result_html = f"<p>Random Forest Classifier Accuracy: {accuracy:.4f}</p>"
            result_html += f"<p>Confusion Matrix:<pre>{confusion_matrix(y_test, preds)}</pre></p>"
            return result_html, chart_info

        # K-nearest-neighbour (Classification example)
        elif operation == "knn":
            temp_df = df.copy()
            for col in temp_df.select_dtypes(include='object').columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col])

            if temp_df.shape[1] < 2:
                result_html = "<p class='text-yellow-500'>Not enough columns for K-Nearest Neighbours (KNN) classification (need at least one feature and a target).</p>"
                return result_html, chart_info

            X = temp_df.iloc[:, :-1].dropna()
            y = temp_df.iloc[:, -1].dropna()

            if X.empty or y.empty or len(y.unique()) < 2:
                result_html = "<p class='text-yellow-500'>Target variable not suitable for classification or insufficient data after dropping NaNs.</p>"
                return result_html, chart_info

            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.empty or y.empty or len(y.unique()) < 2:
                 result_html = "<p class='text-yellow-500'>Insufficient data or target variable issues after alignment and NaN handling.</p>"
                 return result_html, chart_info

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=5) # Example: 5 neighbors
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            
            result_html = f"<p>K-Nearest Neighbours (KNN) Classifier Accuracy: {accuracy:.4f}</p>"
            result_html += f"<p>Confusion Matrix:<pre>{confusion_matrix(y_test, preds)}</pre></p>"
            return result_html, chart_info

        # K-means clustering (Requires numeric data)
        elif operation == "kmeans_clustering":
            numeric_df = df.select_dtypes(include=np.number).dropna()
            if numeric_df.shape[1] > 0 and len(numeric_df) > 1:
                # Example: K=3 clusters
                n_clusters = 3
                if len(numeric_df) < n_clusters:
                    result_html = f"<p class='text-yellow-500'>Not enough data points ({len(numeric_df)}) for {n_clusters} clusters. Please upload a larger dataset or choose fewer clusters.</p>"
                    return result_html, chart_info
                
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                clusters = model.fit_predict(numeric_df)
                
                result_html = f"<p>K-Means Clustering (K={n_clusters}) performed.</p>"
                result_html += f"<p>Cluster assignments (first 10): {clusters[:10].tolist()}...</p>"
                
                # Visualize clusters if possible (e.g., first two components/features)
                if numeric_df.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(x=numeric_df.iloc[:, 0], y=numeric_df.iloc[:, 1], hue=clusters, palette='viridis', ax=ax, legend='full')
                    ax.set_title(f'K-Means Clusters (K={n_clusters})')
                    ax.set_xlabel(numeric_df.columns[0])
                    ax.set_ylabel(numeric_df.columns[1])
                    result_html += render_plot(fig)
                elif numeric_df.shape[1] == 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(x=numeric_df.iloc[:, 0], hue=clusters, palette='viridis', multiple='stack', ax=ax, legend=True)
                    ax.set_title(f'K-Means Clusters (K={n_clusters}) on {numeric_df.columns[0]}')
                    ax.set_xlabel(numeric_df.columns[0])
                    ax.set_ylabel('Count')
                    result_html += render_plot(fig)
            else:
                result_html = "<p class='text-yellow-500'>No numeric data or insufficient data for K-Means Clustering.</p>"
            return result_html, chart_info

        # Linear discriminant analysis (LDA) (Classification example, requires labeled data)
        elif operation == "lda":
            temp_df = df.copy()
            for col in temp_df.select_dtypes(include='object').columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col])

            if temp_df.shape[1] < 2:
                result_html = "<p class='text-yellow-500'>Not enough columns for Linear Discriminant Analysis (LDA) (need at least one feature and a target).</p>"
                return result_html, chart_info

            X = temp_df.iloc[:, :-1].dropna()
            y = temp_df.iloc[:, -1].dropna()

            if X.empty or y.empty or len(y.unique()) < 2: # LDA needs at least 2 classes
                result_html = "<p class='text-yellow-500'>Target variable not suitable for classification (need at least 2 classes) or insufficient data after dropping NaNs.</p>"
                return result_html, chart_info

            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.empty or y.empty or len(y.unique()) < 2:
                 result_html = "<p class='text-yellow-500'>Insufficient data or target variable issues after alignment and NaN handling for LDA.</p>"
                 return result_html, chart_info

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Ensure the number of components is less than or equal to min(n_classes - 1, n_features)
            n_components = min(len(y.unique()) - 1, X.shape[1])
            if n_components < 1:
                result_html = "<p class='text-yellow-500'>Not enough unique classes or features for LDA.</p>"
                return result_html, chart_info
            
            model = LinearDiscriminantAnalysis(n_components=n_components)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            
            result_html = f"<p>Linear Discriminant Analysis (LDA) Classifier Accuracy: {accuracy:.4f}</p>"
            result_html += f"<p>Confusion Matrix:<pre>{confusion_matrix(y_test, preds)}</pre></p>"
            return result_html, chart_info

        # Logistic regression (Binary classification example)
        elif operation == "logistic_regression":
            temp_df = df.copy()
            for col in temp_df.select_dtypes(include='object').columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col])

            if temp_df.shape[1] < 2:
                result_html = "<p class='text-yellow-500'>Not enough columns for Logistic Regression (need at least one feature and a target).</p>"
                return result_html, chart_info

            X = temp_df.iloc[:, :-1].dropna()
            y = temp_df.iloc[:, -1].dropna()

            if X.empty or y.empty or len(y.unique()) != 2: # Logistic regression typically for binary classification
                result_html = "<p class='text-yellow-500'>Target variable not suitable for binary classification (must have exactly 2 unique classes) or insufficient data after dropping NaNs.</p>"
                return result_html, chart_info
            
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.empty or y.empty or len(y.unique()) != 2:
                 result_html = "<p class='text-yellow-500'>Insufficient data or target variable issues after alignment and NaN handling for Logistic Regression.</p>"
                 return result_html, chart_info

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is a good choice for small datasets
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            
            result_html = f"<p>Logistic Regression Classifier Accuracy: {accuracy:.4f}</p>"
            result_html += f"<p>Confusion Matrix:<pre>{confusion_matrix(y_test, preds)}</pre></p>"
            return result_html, chart_info

        # Support Vector Machine (SVM) (Classification example)
        elif operation == "svm":
            temp_df = df.copy()
            for col in temp_df.select_dtypes(include='object').columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col])

            if temp_df.shape[1] < 2:
                result_html = "<p class='text-yellow-500'>Not enough columns for Support Vector Machine (SVM) (need at least one feature and a target).</p>"
                return result_html, chart_info

            X = temp_df.iloc[:, :-1].dropna()
            y = temp_df.iloc[:, -1].dropna()

            if X.empty or y.empty or len(y.unique()) < 2:
                result_html = "<p class='text-yellow-500'>Target variable not suitable for classification or insufficient data after dropping NaNs.</p>"
                return result_html, chart_info
            
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.empty or y.empty or len(y.unique()) < 2:
                 result_html = "<p class='text-yellow-500'>Insufficient data or target variable issues after alignment and NaN handling for SVM.</p>"
                 return result_html, chart_info

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = SVC(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            
            result_html = f"<p>Support Vector Machine (SVM) Classifier Accuracy: {accuracy:.4f}</p>"
            result_html += f"<p>Confusion Matrix:<pre>{confusion_matrix(y_test, preds)}</pre></p>"
            return result_html, chart_info

        # Model evaluation (General concept, specific metrics are shown for each model)
        elif operation == "model_evaluation":
            result_html = "<p>Model evaluation assesses how well a machine learning model performs. Common metrics include accuracy, precision, recall, F1-score, RMSE, R-squared, and confusion matrices, many of which are shown with individual model results.</p>"
            return result_html, chart_info

        # Model assumption (Conceptual, assumptions vary by model)
        elif operation == "model_assumptions":
            result_html = "<p>Model assumptions are conditions that should ideally be met for a statistical or machine learning model to produce reliable results. For example, Linear Regression assumes linearity, independence of errors, normality of residuals, and homoscedasticity.</p>"
            result_html += "<p><em>Assumptions are model-specific and require detailed checks.</em></p>"
            return result_html, chart_info

        # 📉 PCA and Related (1):
        # Principal Component Analysis (PCA)
        elif operation == "pca":
            numeric_df = df.select_dtypes(include=np.number).dropna()
            if numeric_df.shape[1] >= 2:
                pca = PCA(n_components=min(numeric_df.shape[1], 2)) # Extract up to 2 components for visualization
                principal_components = pca.fit_transform(numeric_df)
                
                result_html = "<p>Principal Component Analysis (PCA) performed:</p>"
                result_html += f"<p>Explained Variance Ratio (first {pca.n_components} components): {pca.explained_variance_ratio_.tolist()}</p>"
                result_html += f"<p>Total Explained Variance: {pca.explained_variance_ratio_.sum():.4f}</p>"
                
                # Visualize PCA results
                if pca.n_components >= 2:
                    pc_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pc_df, ax=ax)
                    ax.set_title('PCA: First Two Principal Components')
                    ax.set_xlabel('Principal Component 1')
                    ax.set_ylabel('Principal Component 2')
                    result_html += render_plot(fig)
                elif pca.n_components == 1:
                    result_html += "<p>Only one principal component extracted. Cannot visualize as a scatter plot.</p>"
                    result_html += f"<p>First Principal Component (first 10 values): {principal_components[:10].flatten().tolist()}</p>"
            else:
                result_html = "<p class='text-yellow-500'>Need at least two numeric columns for Principal Component Analysis (PCA).</p>"
            return result_html, chart_info

        else:
            result_html = "<p class='text-red-500'>Function not implemented yet!</p>"
            return result_html, chart_info

    except Exception as e:
        result_html = f"<p class='text-red-500'>An error occurred during operation '{operation}': {str(e)}</p>"
        return result_html, chart_info