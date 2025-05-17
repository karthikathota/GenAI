# Car Resale Value Predictor

A machine learning-based tool that predicts the resale value of cars based on various features such as make, model, year, condition, mileage, and original price.

## Features

- Predicts car resale values using a Random Forest Regressor model
- Supports multiple car makes and models
- Handles both structured input and natural language processing
- Validates input parameters for accuracy
- Provides detailed error messages for invalid inputs

## Supported Car Makes

- Toyota
- Honda
- Ford
- Chevrolet
- Nissan
- BMW
- Mercedes
- Hyundai
- Kia
- Jeep

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- FastMCP

## Installation

1. Clone the repository
2. Install the required dependencies:
3. Perform the Following steps:-

```bash
# Create a new directory for our project
uv init sale
cd sale

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx pandas numpy scikit-learn

```

```bash
pip install pandas numpy scikit-learn fastmcp
```

## Usage

### Using Structured Input

```python
from Sale import estimate_resale_value

# Example with structured input
result = estimate_resale_value(
    make="Toyota",
    model_name="Camry",
    year=2020,
    age=4,
    mileage=50000,
    condition="Good",
    original_price=25000
)
print(result)
```

### Using Natural Language Input

```python
from Sale import estimate_resale_value

# Example with natural language input
result = estimate_resale_value(
    prompt="What's the resale value of a 2020 Toyota Camry in Good condition with 50,000 mileage, originally bought for $25,000?"
)
print(result)
```

## Input Parameters

- `make`: Car manufacturer (e.g., "Toyota", "Honda")
- `model_name`: Car model (e.g., "Camry", "Civic")
- `year`: Manufacturing year (2000-2024)
- `age`: Age of the car in years
- `mileage`: Total mileage (must be positive)
- `condition`: Car condition ("Poor", "Fair", "Good", "Excellent")
- `original_price`: Original purchase price (must be positive)

## Model Details

The prediction model uses:

- Random Forest Regressor with 100 estimators
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Pre-trained on a comprehensive dataset of car resale values

## Error Handling

The tool provides clear error messages for:

- Invalid year range
- Invalid condition values
- Negative mileage
- Non-positive original price
- Missing required parameters
- Parsing errors in natural language input

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
