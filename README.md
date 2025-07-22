# Gold Price Forecasting using ARIMA and SARIMA

This project focuses on forecasting the price of gold using time series analysis techniques, specifically ARIMA and SARIMA models. The gold price data was obtained from the yfinance library. *An interactive web application has also been developed using Streamlit to demonstrate the forecasting capabilities.*

## Table of Contents

- Project Overview
- Live Demo 
- Dataset
- Methodology
- Exploratory Data Analysis (EDA)
- Seasonality Decomposition
- Stationarity
- ACF and PACF plots to determine model orders
- Model Building (ARIMA & SARIMA)
- Forecasting](#forecasting)
- Results](#results)
- Model saving using pickle 
- Future Work
- Installation
- Usage
- Contributing
- License
- Contact

## Project Overview

The primary goal of this project is to build robust time series models to predict future gold prices. Understanding and predicting gold price movements can be valuable for investors, financial analysts, and economists. This project demonstrates a typical time series workflow, from data acquisition and preprocessing to model training and evaluation. *Additionally, a user-friendly Streamlit application allows for interactive exploration of the model's forecasts.*

## Live Demo ✨

Experience the gold price forecasting model in action!

- Choose between ARIMA and SARIMA models
- Forecast up to 36 months into the future
- Visualize forecast with **confidence intervals**
- View model summary and raw historical data

 **You can interact with the deployed application here:**

https://timeseriessarimagoldmodel-77kmxdm9mehjhyaporyrlq.streamlit.app/


Note: The application might take a few moments to load on the first visit.

## Dataset

The gold price data was sourced using the yfinance library, which provides access to historical market data from Yahoo Finance.

- *Data Source:* yfinance
- *Ticker:* GC=F 
- *Time Period:* start = "2005-01-01", end = "2025-07-01", interval = 1month


## Methodology

### Exploratory Data Analysis (EDA)

Initial data exploration involved:
- Visualizing the gold price series over time.
- Checking for trends, seasonality, and cycles.
- Analyzing summary statistics.

### Seasonality Decomposition

The time series was decomposed into its trend, seasonal, and residual components to better understand underlying patterns. This was likely achieved using techniques like statsmodels.tsa.seasonal.seasonal_decompose.

### Stationarity

For ARIMA and SARIMA models, it's crucial for the time series to be stationary (constant mean, variance, and autocorrelation over time). Methods used to achieve stationarity included:
- *Differencing:* first-order differencing
- *Augmented Dickey-Fuller (ADF) Test:* Used to statistically confirm stationarity after differencing.

## ADF Test Result Sample Output

ADF Statistic: -1.56
p-value: 0.51
Conclusion: Time series is **non-stationary**. Differencing is required.

### Model Building (ARIMA & SARIMA)

*ARIMA (AutoRegressive Integrated Moving Average) Model:*
- *Parameters:* (p, d, q) where:
    - p: Order of the AR part.
    - d: Order of differencing.
    - q: Order of the MA part.
- *Parameter Selection:* Used ACF/PACF plots.

*SARIMA (Seasonal AutoRegressive Integrated Moving Average) Model:*
- *Parameters:* (p, d, q)(P, D, Q, S) where:
    - (p, d, q): Non-seasonal parameters.
    - (P, D, Q): Seasonal parameters.
    - S: Seasonal period
- *Parameter Selection:* seasonal ACF/PACF plots.

## Model Details

| Model  | AIC / BIC | Seasonality | Stationarity | Forecast Horizon |
| ------ | --------- | ----------- | ------------ | ---------------- |
| ARIMA  | Evaluated | ❌ No        | Differenced  | Short-term       |
| SARIMA | Evaluated | ✅ Yes       | Differenced  | Seasonal-aware  

### Forecasting

Once the models were trained, they were used to forecast gold prices for a specified future period.

## Results


- *Model Performance:*
    - Evaluation metrics used is AIC and BIC.
    - Compared the performance of ARIMA vs. SARIMA and Observed, the SARIMA is performing well based on AIC and BIC
  

*Key Findings:*
- SARIMA model demonstrated superior performance due to the some seasonality observed in gold prices."
- The model predicts a slight decrease in gold prices over the next 3 months, and then its increasing.

## Future Work

- *Incorporate External Features:* Explore the impact of macroeconomic indicators (e.g., interest rates, inflation, USD index, geopolitical events) as exogenous variables in more advanced models (e.g., SARIMAX).
- *Alternative Models:* Experiment with other time series models such as Prophet, LSTM neural networks, or XGBoost for time series.
- *Hyperparameter Optimization:* Implement more rigorous hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV) for ARIMA/SARIMA models.
- *Ensemble Modeling:* Combine forecasts from multiple models to potentially improve accuracy.


## 🗃️ Repository Structure

```

📁 gold-price-forecasting/
├── app.py                   # Streamlit app
├── arima\_gold\_model.pkl     # Trained ARIMA model (auto-generated)
├── sarima\_gold\_model.pkl    # Trained SARIMA model (auto-generated)
├── model\_training.ipynb     # Main analysis & modeling notebook (optional)
├── requirements.txt         # Required packages
└── README.md                # Project overview



## 🧰 Technologies Used

* `Python`
* `Pandas`, `Matplotlib`, `Statsmodels`
* `yfinance` for data fetching
* `Streamlit` for UI
* `Pickle` for model persistence

## Installation

To run this project, you'll need Python and the following libraries.

```bash
# Clone the repository
git clone https://github.com/Mohammadi-Nilofer/Time_Series_Sarima_Gold_model
cd gold-price-forecasting


# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels yfinance scikit-learn

## 📌 `requirements.txt`

If you want to deploy or share the project, use this for your `requirements.txt`:

```

streamlit
pandas
matplotlib
statsmodels
yfinance
python-dateutil


## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

Developed by **\Mohammadi Nilofer**
📧 niloferm7@yahoo.com
🌐 https://github.com/Mohammadi-Nilofer/Time_Series_Sarima_Gold_model
