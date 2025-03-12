# Car Price Prediction Project Evolution (Assignments of Machine Learning Course at AIT)

This repository contains three iterations of a car price prediction project, each building upon the previous version with enhanced features and technologies. These are assignments of the Machine Learning Course at AIT.

## Project Overview

### A1: Basic Regression Model
The initial version implements a Random Forest Regressor to predict car prices based on various features.

**Key Features:**
- Dash web interface
- Random Forest Regression model
- Basic feature preprocessing
- Real-time predictions
- Local model deployment

### A2: Docker & MLflow Integration
The second version enhances the project with containerization and model tracking capabilities.

**Key Features:**
- Dockerized application
- MLflow integration for model versioning
- Enhanced UI with tabbed interface
- Support for model comparison
- Improved feature preprocessing
- Containerized deployment

### A3: Classification & Testing
The final version transforms the problem into a classification task with comprehensive testing.

**Key Features:**
- Four-category price classification
- Remote MLflow model registry
- Automated unit testing
- Enhanced feature normalization
- Staging environment support
- Version control (v1.0.37)

## Technology Stack Evolution

### Infrastructure
- **A1:** Local development
- **A2:** Docker containerization
- **A3:** Docker + Remote MLflow registry

### Model Development
- **A1:** Random Forest Regression
- **A2:** Enhanced Regression with MLflow tracking
- **A3:** Multi-class Classification with remote model registry

### Testing & Quality
- **A1:** Manual testing
- **A2:** Basic validation
- **A3:** Automated unit testing

## Common Features Across Versions

All versions share these core features:
- Web-based interface using Dash
- Seven input features:
  - Model year (1886-2023)
  - Fuel type (Diesel/Petrol)
  - Seller type (Individual/Dealer/Trustmark Dealer)
  - Transmission type (Manual/Automatic)
  - Owner history (First/Second/Third/Fourth & Above)
  - Engine size (CC)
  - Maximum power (BHP)

## Project Structure

```
Assignments/
├── A1/                        # Basic Regression Model
│   ├── README.md
│   ├── dash_app.py
│   ├── model/
│   └── requirements.txt
│
├── A2/                        # Docker & MLflow Integration
│   ├── README.md
│   ├── MLflow/
│   ├── code/
│   ├── dash.Dockerfile
│   └── docker-compose.yaml
│
└── A3/                        # Classification & Testing
    ├── README.md
    ├── code/
    │   ├── main.py
    │   └── test_app_callbacks.py
    ├── dash.Dockerfile
    └── docker-compose.yaml
```

## Key Improvements Through Versions

1. **Model Evolution**
   - A1: Basic regression predictions
   - A2: Enhanced regression with versioning
   - A3: Sophisticated classification with categories

2. **Deployment Strategy**
   - A1: Local deployment
   - A2: Containerized deployment
   - A3: Container orchestration with remote model registry

3. **Code Quality**
   - A1: Basic structure
   - A2: Modular organization
   - A3: Test-driven development

4. **User Experience**
   - A1: Single model interface
   - A2: Model comparison interface
   - A3: Categorical predictions with confidence

## Getting Started

Each version has its own README with specific setup instructions. Generally:

1. **A1:** Run locally with Python
   ```bash
   pip install -r requirements.txt
   python dash_app.py
   ```

2. **A2:** Build and run with Docker
   ```bash
   docker-compose up --build
   ```

3. **A3:** Build, test, and run with Docker
   ```bash
   docker-compose up --build
   python -m pytest code/test_app_callbacks.py
   ```

## Technical Details

- **Programming Language:** Python 3.10.12
- **Web Framework:** Dash
- **ML Libraries:** scikit-learn, MLflow
- **Containerization:** Docker
- **Testing:** pytest
- **Model Registry:** MLflow remote server

## Future Improvements

Potential areas for further development:
1. Enhanced feature engineering
2. Additional model architectures
3. A/B testing capabilities
4. Real-time model monitoring
5. Performance optimization
6. Extended test coverage
