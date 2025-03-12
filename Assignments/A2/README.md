# Car Price Prediction with MLflow and Docker

An enhanced version of the car price prediction application, now containerized with Docker and featuring MLflow for model tracking and management.

## Project Structure

```
├── MLflow/                     # MLflow service configuration
│   ├── docker-compose.yml      # MLflow service orchestration
│   ├── mlflow.Dockerfile       # MLflow server configuration
│   └── python.Dockerfile       # Python environment for MLflow
├── code/                       # Main application code
│   ├── Images/                 # Image assets
│   ├── data/                   # Dataset directory
│   ├── model/                  # MLflow model artifacts
│   ├── old_model/             # Previous model version
│   ├── mlruns/                # MLflow experiment tracking
│   ├── main.py                # Main Dash application
│   └── TEST.py                # Testing module
├── dash.Dockerfile            # Dash application container config
└── docker-compose.yaml        # Main service orchestration
```

## Features

- Dockerized application for consistent deployment
- MLflow integration for model versioning and tracking
- Enhanced UI with tabbed interface for model comparison
- Support for both new and old model versions
- Comprehensive input validation and preprocessing
- Real-time price predictions

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
3. Click 'Predict' to get the estimated price

## Model Information

### New Model (MLflow)
- Normalized feature scaling
- Enhanced sensitivity to feature variations
- Tracked using MLflow for version control

### Old Model (Legacy)
- Standard feature preprocessing
- Higher accuracy but less sensitive to minor variations

## Development

- MLflow UI available for experiment tracking
- Testing module included for model validation
- Docker environment ensures consistent development experience

## Notes

- The new model emphasizes sensitivity to feature variations
- Both models available for comparison through the tabbed interface
- All services are containerized for easy deployment
