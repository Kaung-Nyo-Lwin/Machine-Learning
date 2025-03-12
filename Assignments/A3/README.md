# Car Price Classification with MLflow Integration

A machine learning application that classifies car prices into categories (Low, Medium, High, Very High) based on various features. This version includes MLflow integration for model tracking and automated testing.

## Project Structure

```
├── code/                       # Main application code
│   ├── data/                   # Dataset directory
│   ├── model/                  # MLflow model artifacts
│   ├── models/                 # Model storage
│   ├── main.py                # Main Dash application
│   └── test_app_callbacks.py    # Unit tests for app callbacks
├── dash.Dockerfile            # Dash application container config
└── docker-compose.yaml        # Service orchestration
```

## Features

- Car price classification into four categories:
  - Low
  - Medium
  - High
  - Very High
- MLflow integration with remote model registry
- Automated unit testing
- Dockerized deployment
- Feature normalization and preprocessing
- Real-time predictions

## Requirements

- Docker and Docker Compose
- Python 3.10.12
- Required Python packages (installed via Docker):
  - dash
  - pandas
  - dash_bootstrap_components
  - numpy
  - scikit-learn
  - mlflow
  - pytest (for testing)

## Installation & Setup

1. Clone this repository
2. Build and start the services:
   ```bash
   docker-compose up --build
   ```

## Usage

1. Access the application at `http://localhost:9009`
2. Input the following car features:
   - Model year (1886-2023)
   - Fuel type (Diesel/Petrol)
   - Seller type (Individual/Dealer/Trustmark Dealer)
   - Transmission type (Manual/Automatic)
   - Owner history (First/Second/Third/Fourth & Above)
   - Engine size (CC)
   - Maximum power (BHP)
3. Click 'Predict' to get the price category

## Model Information

### Classification Model v1.0.37
- Classifies car prices into four categories
- Features normalized using StandardScaler
- Remote model registry integration via MLflow
- Automatic model versioning and staging

## MLflow Integration

- Remote MLflow tracking server at `mlflow.ml.brain.cs.ait.ac.th`
- Model versioning and staging support
- Experiment tracking under `st125066-a3`
- Model name: `st125066-a3-model`

## Testing

The application includes unit tests for:
- Model output shape validation
- Model coefficient validation
- Prediction functionality

Run tests using:
```bash
python -m pytest test_app_callbacks.py
```

## Development Notes

- Model is served from staging environment
- Automated model loading from remote registry
- Comprehensive input validation and preprocessing
- Docker environment for consistent deployment
