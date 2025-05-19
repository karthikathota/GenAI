# Weather Predictor

A Python-based weather information service that provides real-time weather alerts and forecasts using the National Weather Service (NWS) API.

## Features

- Get active weather alerts for any US state
- Retrieve detailed weather forecasts for specific locations using latitude and longitude
- Real-time data from the National Weather Service API
- Asynchronous API calls for better performance

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd WeatherPredictor
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The Weather Predictor provides two main functions:

### 1. Get Weather Alerts

Retrieve active weather alerts for any US state using its two-letter code:

```python
from weather.weather import get_alerts

# Example: Get alerts for California
alerts = await get_alerts("CA")
print(alerts)
```

### 2. Get Weather Forecast

Get detailed weather forecasts for a specific location using latitude and longitude:

```python
from weather.weather import get_forecast

# Example: Get forecast for San Francisco (approximately)
forecast = await get_forecast(37.7749, -122.4194)
print(forecast)
```

## API Reference

### `get_alerts(state: str) -> str`

Retrieves active weather alerts for a specified US state.

- **Parameters:**
  - `state`: Two-letter US state code (e.g., "CA", "NY")
- **Returns:**
  - Formatted string containing active weather alerts
  - Returns "No active alerts" if none are found

### `get_forecast(latitude: float, longitude: float) -> str`

Retrieves weather forecast for a specific location.

- **Parameters:**
  - `latitude`: Location's latitude (float)
  - `longitude`: Location's longitude (float)
- **Returns:**
  - Formatted string containing detailed weather forecast
  - Includes temperature, wind conditions, and detailed forecast

## Error Handling

The service includes built-in error handling for:

- API connection issues
- Invalid state codes
- Invalid coordinates
- Missing or malformed data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- National Weather Service (NWS) for providing the weather data API
- FastMCP for the server implementation
