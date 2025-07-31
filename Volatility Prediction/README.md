# Volatility Forecasting with GARCH Models

[cite_start]This project, based on the report "Volatility Modeling and Forecasting using Time Series Models" by Aryanil Kundu from the Indian Statistical Institute, demonstrates the application of GARCH(1,1) models to forecast financial volatility on the Indian stock market[cite: 6].

## 📜 Project Overview

[cite_start]The core objective is to model and forecast the volatility of several major Indian stocks from different sectors[cite: 13, 47]. [cite_start]The analysis recognizes that financial return series exhibit time-varying volatility and volatility clustering—periods of high volatility tend to be followed by more high volatility, and calm periods are followed by calm[cite: 14, 42, 57].

[cite_start]This project uses the **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** framework to capture these dynamics effectively[cite: 14]. [cite_start]The findings show that even a simple GARCH(1,1) model can successfully predict volatility patterns and is a valuable tool for financial risk management[cite: 19, 213].

---

## 🔑 Key Features

* [cite_start]**Model**: A **GARCH(1,1)** model is used to capture the dynamic structure of volatility[cite: 16, 49]. The model is defined as:
    $$\sigma_{t}^{2}=\omega+\alpha_{1}\epsilon_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}$$
    [cite_start]where $\sigma_{t}^{2}$ is the conditional variance, $\epsilon_{t-1}^{2}$ is the past squared shock (ARCH term), and $\sigma_{t-1}^{2}$ is the past variance (GARCH term)[cite: 80, 81].
* [cite_start]**Dataset**: The analysis uses daily closing prices for major Indian stocks and the NSEI index from January 1, 2019, to December 31, 2024[cite: 94, 95].
* [cite_start]**Stocks Analyzed**: The model was applied to a diverse set of equities[cite: 149]:
    * [cite_start]NSEI (Nifty 50 Index) [cite: 150]
    * [cite_start]TCS (Tata Consultancy Services) [cite: 158]
    * [cite_start]ONGC (Oil and Natural Gas Corporation) [cite: 169]
    * [cite_start]SBI (State Bank of India) [cite: 177]
    * [cite_start]RELIANCE (Reliance Industries) [cite: 189]
* **Validation**: Model performance is validated through rigorous diagnostics, including:
    * [cite_start]**Residual Analysis**: Checking the Autocorrelation Function (ACF) of standardized residuals to ensure no remaining conditional heteroskedasticity[cite: 146, 212].
    * [cite_start]**Error Metrics**: Quantitative evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) against realized volatility[cite: 197].

---

## 🛠️ Methodology

The project follows a systematic time series analysis workflow:

1.  [cite_start]**Data Preparation**: Raw daily closing prices are converted into **log returns** ($r_t = \log(P_t/P_{t-1})$) to achieve stationarity, a key requirement for time series modeling[cite: 132, 134, 137].
2.  [cite_start]**Train-Test Split**: The historical data is divided into a training set for model estimation and a testing set for out-of-sample forecasting[cite: 92, 93].
3.  [cite_start]**Rolling Forecast**: To predict volatility for each day in the test set, the GARCH(1,1) model is retrained using all available historical data up to that point[cite: 143]. [cite_start]This ensures the model adapts to the most recent volatility trends[cite: 150, 158].
4.  [cite_start]**Evaluation**: The forecasted volatility is compared against **realized volatility**, which is calculated as the rolling standard deviation of log returns over a 15-day window[cite: 144, 159, 170, 178, 190].

---

## 📊 Results Summary

The GARCH(1,1) model proved effective in capturing the core characteristics of financial volatility for all analyzed stocks.

* [cite_start]**Volatility Clustering**: The forecasts successfully mirrored the realized volatility patterns, where periods of high and low volatility were clustered together[cite: 156, 166, 175, 186, 195].
* [cite_start]**Model Significance**: The model coefficients ($\alpha_1$ and $\beta_1$) were statistically significant across all stocks, and the stationarity condition ($\alpha_1 + \beta_1 < 1$) was met[cite: 153, 162, 173, 193].
* [cite_start]**Residuals**: The ACF plots of standardized residuals for all models showed no significant autocorrelation, confirming that the models successfully captured the conditional heteroskedasticity[cite: 155, 165, 174, 185].

### Performance Metrics

[cite_start]The forecasting accuracy was measured over a 1-year out-of-sample period, with the following results[cite: 207]:

| Stock    | MAE    | RMSE   | MAPE (%) |
| :------- | :----- | :----- | :------- |
| NSEI     | 0.0015 | 0.0020 | 25.52    |
| TCS      | 0.0023 | 0.0027 | 21.74    |
| **ONGC** | 0.0035 | 0.0050 | **19.86**|
| RELIANCE | 0.0025 | 0.0030 | 24.21    |
| SBI      | 0.0031 | 0.0049 | 21.77    |

[cite_start]The model for **ONGC** was particularly accurate, achieving the lowest MAPE[cite: 208]. [cite_start]For some stocks like TCS and SBI, the model tended to overestimate low-volatility regions but was still effective for differentiating between high and low-volatility regimes[cite: 167, 187, 188].

---

## 💻 Code and Usage

[cite_start]The Python code used for this analysis is included in the repository[cite: 223]. [cite_start]To forecast the volatility of a different stock, you can modify the `ticker` variable within the main script[cite: 223].

---

## 🚀 Future Work

[cite_start]This project lays the groundwork for several potential extensions[cite: 214]:

* [cite_start]Explore more advanced models like **EGARCH** or **GJR-GARCH** to account for asymmetric volatility responses to positive and negative news[cite: 214].
* [cite_start]Integrate an **ARIMA** model for the mean return process to create a more comprehensive ARIMA-GARCH model[cite: 214].
* [cite_start]Apply the volatility forecasts to practical risk management applications, such as calculating **Value at Risk (VaR)**[cite: 214].