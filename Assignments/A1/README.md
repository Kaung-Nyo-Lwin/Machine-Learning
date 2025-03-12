# Car Price Prediction Web Application

A web-based application that predicts car prices using a Random Forest Regressor model based on various features.

## Overview

This application provides a user-friendly interface for predicting car prices based on 7 key parameters:
- Model year (1886-2023)
- Fuel type (Diesel/Petrol)
- Seller type (Individual/Dealer/Trustmark Dealer)
- Transmission type (Manual/Automatic)
- Owner history (First/Second/Third/Fourth & Above)
- Engine size (CC)
- Maximum power (BHP)

## Project Structure

```
├── dash_app.py         # Main application file with Dash web interface
├── model/              # Directory containing the trained model
│   └── car-price.model # Trained Random Forest Regressor model
└── requirements.txt    # Project dependencies
```

## Requirements

The following packages are required to run the application:
- dash
- gunicorn
- pandas
- dash_bootstrap_components
- numpy
- scikit-learn

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python dash_app.py
   ```
2. Open your web browser and navigate to the local server address
3. Input the required car features using the dropdown menus and input fields:
   - Select the model year from the dropdown (1886-2023)
   - Choose the fuel type (Diesel/Petrol)
   - Select the seller type
   - Choose the transmission type
   - Specify the owner history
   - Enter the engine size in CC
   - Input the maximum power in BHP
4. Click the 'Predict' button to get the estimated car price

## Features

- Interactive web interface built with Dash
- Real-time price predictions using Random Forest Regressor
- Comprehensive input validation and data preprocessing
- Support for various car features with dropdown selections
- Pre-trained machine learning model for accurate predictions

## Model Information

The application uses a pre-trained Random Forest Regressor model stored in `model/car-price.model`. The model processes the input features through several steps:
1. Data preprocessing and normalization
2. Feature encoding for categorical variables
3. Prediction using the trained model

