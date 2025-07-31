# Volatility Forecasting with GARCH Models

This project, based on the report "Volatility Modeling and Forecasting using Time Series Models" by Aryanil Kundu from the Indian Statistical Institute, demonstrates the application of GARCH(1,1) models to forecast financial volatility on the Indian stock market.

## 📜 Project Overview

The core objective is to model and forecast the volatility of several major Indian stocks from different sectors.The analysis recognizes that financial return series exhibit time-varying volatility and volatility clustering—periods of high volatility tend to be followed by more high volatility, and calm periods are followed by calm.

This project uses the **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** framework to capture these dynamics effectively.The findings show that even a simple GARCH(1,1) model can successfully predict volatility patterns and is a valuable tool for financial risk management.

---

## 🔑 Key Features

* **Model**: A **GARCH(1,1)** model is used to capture the dynamic structure of volatility. The model is defined as:
    $$\sigma_{t}^{2}=\omega+\alpha_{1}\epsilon_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}$$
    where $\sigma_{t}^{2}$ is the conditional variance, $\epsilon_{t-1}^{2}$ is the past squared shock (ARCH term), and $\sigma_{t-1}^{2}$ is the past variance (GARCH term).
* **Dataset**: The analysis uses daily closing prices for major Indian stocks and the NSEI index from January 1, 2019, to December 31, 2024.
* **Stocks Analyzed**: The model was applied to a diverse set of equities:
    * NSEI (Nifty 50 Index)
    * TCS (Tata Consultancy Services)
    * ONGC (Oil and Natural Gas Corporation) 
    * SBI (State Bank of India) 
    * RELIANCE (Reliance Industries) 
* **Validation**: Model performance is validated through rigorous diagnostics, including:
  * **Residual Analysis**: Checking the Autocorrelation Function (ACF) of standardized residuals to ensure no remaining conditional heteroskedasticity.
    * **Error Metrics**: Quantitative evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) against realized volatility.

---

## 🛠️ Methodology

The project follows a systematic time series analysis workflow:

1.  **Data Preparation**: Raw daily closing prices are converted into **log returns** ($r_t = \log(P_t/P_{t-1})$) to achieve stationarity, a key requirement for time series modeling.
2.  **Train-Test Split**: The historical data is divided into a training set for model estimation and a testing set for out-of-sample forecasting.
3.  **Rolling Forecast**: To predict volatility for each day in the test set, the GARCH(1,1) model is retrained using all available historical data up to that point. This ensures the model adapts to the most recent volatility trends.
4.  **Evaluation**: The forecasted volatility is compared against **realized volatility**, which is calculated as the rolling standard deviation of log returns over a 15-day window.

---

## 📊 Results Summary

The GARCH(1,1) model proved effective in capturing the core characteristics of financial volatility for all analyzed stocks.

* **Volatility Clustering**: The forecasts successfully mirrored the realized volatility patterns, where periods of high and low volatility were clustered together.
* **Model Significance**: The model coefficients ($\alpha_1$ and $\beta_1$) were statistically significant across all stocks, and the stationarity condition ($\alpha_1 + \beta_1 < 1$) was met.
* **Residuals**: The ACF plots of standardized residuals for all models showed no significant autocorrelation, confirming that the models successfully captured the conditional heteroskedasticity.

### Performance Metrics

The forecasting accuracy was measured over a 1-year out-of-sample period, with the following results:

| Stock    | MAE    | RMSE   | MAPE (%) |
| :------- | :----- | :----- | :------- |
| NSEI     | 0.0015 | 0.0020 | 25.52    |
| TCS      | 0.0023 | 0.0027 | 21.74    |
| **ONGC** | 0.0035 | 0.0050 | **19.86**|
| RELIANCE | 0.0025 | 0.0030 | 24.21    |
| SBI      | 0.0031 | 0.0049 | 21.77    |

The model for **ONGC** was particularly accurate, achieving the lowest MAPE. For some stocks like TCS and SBI, the model tended to overestimate low-volatility regions but was still effective for differentiating between high and low-volatility regimes.

---

## 💻 Code and Usage

The Python code used for this analysis is included in the repository. To forecast the volatility of a different stock, you can modify the `ticker` variable within the main script.

---

## 🚀 Future Work

This project lays the groundwork for several potential extensions:

* Explore more advanced models like **EGARCH** or **GJR-GARCH** to account for asymmetric volatility responses to positive and negative news.
* Integrate an **ARIMA** model for the mean return process to create a more comprehensive ARIMA-GARCH model.
* Apply the volatility forecasts to practical risk management applications, such as calculating **Value at Risk (VaR)**.
