import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

# Function to calculate the confidence interval
def compute_confidence_interval(y_true, y_pred, confidence=0.95):
    mse = mean_squared_error(y_true, y_pred)
    se = sqrt(mse)  # Standard error
    z = 1.96 if confidence == 0.95 else 1.645  # Z-score for 95% or 90%
    margin_of_error = z * se
    point_estimate = np.mean(y_pred)
    lower_bound = point_estimate - margin_of_error
    upper_bound = point_estimate + margin_of_error
    return point_estimate, margin_of_error, lower_bound, upper_bound

# Load the dataset
data = pd.read_csv(r'C:\Users\varsh\Downloads\data_final.csv')

# Convert 'Month' to datetime
data['Month'] = pd.to_datetime(data['Month'])

# Create dummy variables for 'Seasonality Factor'
data = pd.get_dummies(data, columns=['Seasonality Factor'], drop_first=True)

# Filter data by variants
selected_variants_lr = ['XXX11', 'XXX15', 'XXX18','XXXV5','XXXV9']
selected_variants_xgb = ['XXX12', 'XXX13', 'XXX17']
selected_variants_se = ['XXXV1', 'XXXV2', 'XXXV3', 'XXXV4']

filtered_data_lr = data[data['Variant'].isin(selected_variants_lr)]
filtered_data_xgb = data[data['Variant'].isin(selected_variants_xgb)]
filtered_data_se = data[data['Variant'].isin(selected_variants_se)]

# Train Linear Regression Models
linear_models = {}
for variant_lr in filtered_data_lr['Variant']:
    variant_data_lr = filtered_data_lr[filtered_data_lr['Variant'] == variant_lr]
    X_variant_lr = variant_data_lr[['Economic Index', 'Seasonality Factor_Medium', 'Seasonality Factor_Low']]
    y_variant_lr = variant_data_lr['Industry Growth Rate (%)']
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_variant_lr, y_variant_lr, test_size=0.2, shuffle=False)
    
    linear_model = LinearRegression()
    linear_model.fit(X_lr_train, y_lr_train)
    linear_models[variant_lr] = linear_model

# Train XGBoost Models
xgb_models = {}
for variant_xgb in filtered_data_xgb['Variant']:
    variant_data_xgb = filtered_data_xgb[filtered_data_xgb['Variant'] == variant_xgb]
    X_xgb_variant = variant_data_xgb[['Economic Index', 'Seasonality Factor_Medium', 'Seasonality Factor_Low']]
    y_xgb_variant = variant_data_xgb['Industry Growth Rate (%)']
    X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(X_xgb_variant, y_xgb_variant, test_size=0.2, shuffle=False)
    
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_xgb_train, y_xgb_train)
    xgb_models[variant_xgb] = xgb_model

# Train Simple Exponential Smoothing (SES) Models
ses_models = {}
for variant_se in filtered_data_se['Variant']:
    variant_data_se = filtered_data_se[filtered_data_se['Variant'] == variant_se]
    y_se_variant = variant_data_se['Industry Growth Rate (%)']
    ses_model = SimpleExpSmoothing(y_se_variant).fit(smoothing_level=0.2, optimized=False)
    ses_models[variant_se] = ses_model

# Create a results dataframe
results = []

# Calculate confidence intervals for Linear Regression Models
for variant, model in linear_models.items():
    variant_data = filtered_data_lr[filtered_data_lr['Variant'] == variant]
    X_test = variant_data[['Economic Index', 'Seasonality Factor_Medium', 'Seasonality Factor_Low']]
    y_true = variant_data['Industry Growth Rate (%)']
    y_pred = model.predict(X_test)
    
    # Compute 95% and 90% confidence intervals
    point_estimate_95, moe_95, lower_95, upper_95 = compute_confidence_interval(y_true, y_pred, confidence=0.95)
    point_estimate_90, moe_90, lower_90, upper_90 = compute_confidence_interval(y_true, y_pred, confidence=0.90)
    
    results.append([variant, 'Linear Regression', '95%', lower_95, upper_95, point_estimate_95])
    results.append([variant, 'Linear Regression', '90%', lower_90, upper_90, point_estimate_90])

# Calculate confidence intervals for XGBoost Models
for variant, model in xgb_models.items():
    variant_data = filtered_data_xgb[filtered_data_xgb['Variant'] == variant]
    X_test = variant_data[['Economic Index', 'Seasonality Factor_Medium', 'Seasonality Factor_Low']]
    y_true = variant_data['Industry Growth Rate (%)']
    y_pred = model.predict(X_test)
    
    # Compute 95% and 90% confidence intervals
    point_estimate_95, moe_95, lower_95, upper_95 = compute_confidence_interval(y_true, y_pred, confidence=0.95)
    point_estimate_90, moe_90, lower_90, upper_90 = compute_confidence_interval(y_true, y_pred, confidence=0.90)
    
    results.append([variant, 'XGBoost', '95%', lower_95, upper_95, point_estimate_95])
    results.append([variant, 'XGBoost', '90%', lower_90, upper_90, point_estimate_90])

# Calculate confidence intervals for SES Models
for variant, model in ses_models.items():
    y_true = filtered_data_se[filtered_data_se['Variant'] == variant]['Industry Growth Rate (%)']
    y_pred = model.fittedvalues
    
    # Compute 95% and 90% confidence intervals
    point_estimate_95, moe_95, lower_95, upper_95 = compute_confidence_interval(y_true, y_pred, confidence=0.95)
    point_estimate_90, moe_90, lower_90, upper_90 = compute_confidence_interval(y_true, y_pred, confidence=0.90)
    
    results.append([variant, 'SES', '95%', lower_95, upper_95, point_estimate_95])
    results.append([variant, 'SES', '90%', lower_90, upper_90, point_estimate_90])

# Convert the results to a DataFrame
results_df = pd.DataFrame(results, columns=['Variant', 'Model', 'Confidence Interval', 'Lower Bound', 'Upper Bound', 'Point Estimate'])

# Streamlit app
st.title('Model Confidence Intervals and Point Estimates')

# Display the results dataframe
st.write("Confidence Interval Results:")
st.dataframe(results_df)

# Plotting the intervals
for variant in results_df['Variant'].unique():
    variant_data = results_df[results_df['Variant'] == variant]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(variant_data['Confidence Interval'], variant_data['Point Estimate'], 
                 yerr=variant_data['Upper Bound'] - variant_data['Point Estimate'], fmt='o', label=f'{variant}')
    plt.title(f'Confidence Intervals for {variant}')
    plt.xlabel('Confidence Level')
    plt.ylabel('Industry Growth Rate (%)')
    plt.legend()
    plt.grid(True)
    
    # Show the plot in Streamlit
    st.pyplot(plt)

st.success("Deployment is complete!")
