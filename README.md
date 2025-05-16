# Apple stocks predict

---

## 1. Introduction
The stock market plays a critical role in the global economy. Predicting stock prices is a valuable tool that supports investors in making informed decisions and helps institutions manage financial risks effectively.
This project focuses on the collection, processing, and analysis of Apple Inc. (AAPL) stock price data from 2015 to the present using PySpark for scalable data processing. The goal is to clean and prepare the data, compute key technical indicators, and visualize price trends as a foundation for future modeling and prediction.

---

## 2. Project Objectives

- ✅ Collect historical stock price data for AAPL from Yahoo Finance.
- ✅ Process the data using PySpark to ensure scalability and efficiency.
- ✅ Calculate and visualize basic technical indicators (e.g., moving averages).
- ⏳ (To be added) Develop predictive models for stock price forecasting.

---

## 3. Technologies Used

| Tool        | Purpose                                |
|-------------|----------------------------------------|
| `Python`    | Main programming language              |
| `PySpark`   | Big data processing                    |
| `yfinance`  | Download historical data from Yahoo Finance |
| `matplotlib`| Data visualization                     |
| `pandas`    | Lightweight data manipulation          |

---

## 4. Project Workflow
### 🔍  Data Source and Collection
The historical stock data of Apple is retrieved from Yahoo Finance, starting from January 1, 2015 to the current date. The dataset includes daily stock metrics such as Open, High, Low, Close, Volume, and Adjusted Close. After downloading, the data is structured into a tabular format with a clean and standardized schema, ready for transformation and analysis.
### ⚙️ Data Preparation and Cleaning
Before computing indicators, the data undergoes preprocessing steps to ensure compatibility and efficiency within the PySpark environment. This includes:
  - Converting the Date column to a proper date type to enable time-based operations.
  - Selecting only the relevant columns (Date and Close) to reduce memory usage and simplify computations.
  - Sorting the dataset chronologically to maintain time series integrity.
This preparation ensures that large volumes of stock data can be processed efficiently, maintaining accuracy and speed.
### 📈 30-Day Simple Moving Average (MA30)
The 30-day Simple Moving Average (SMA) is a fundamental trend-following indicator in technical analysis. It reflects the average closing price of the stock over the past 30 trading days. This smoothing technique helps eliminate short-term volatility and highlights the underlying trend.
In this analysis, MA30 is calculated using a rolling window method. The indicator is then visualized alongside the actual closing prices to show how the market behaves in relation to its recent average.
Insight:
When the closing price crosses above the MA30 line, it may suggest an upward momentum. Conversely, when it drops below, it could indicate a bearish trend.
### 📊 14-Day Relative Strength Index (RSI14)
The Relative Strength Index (RSI) is a momentum oscillator used to evaluate whether a stock is overbought or oversold based on recent price changes. In this project, a 14-day RSI is calculated.
  - RSI > 70 typically signals that the stock may be overbought.
  - RSI < 30 indicates a potential oversold condition.
The RSI is derived by averaging recent gains and losses over a 14-day window and applying the RSI formula. It provides a normalized value between 0 and 100 that helps investors assess market sentiment and potential reversal points.
### 📉 Visualization of Indicators
Two key charts are generated to support the analysis:
#### 1.Price vs. MA30 Chart
![Image](https://github.com/user-attachments/assets/e4085327-e9ea-477e-9b46-4e0ffd928f9c)
Displays the closing price of AAPL stock over time with the 30-day moving average (MA30).
  - MA30 smooths short-term fluctuations and highlights trends.
  - Price above MA30 → uptrend; price below MA30 → possible downtrend.
  - Useful for identifying trend direction and potential entry/exit points.
#### 2.RSI Chart
![Image](https://github.com/user-attachments/assets/711dead9-dcac-42a2-bfe0-01e9da7d9936)
Shows the 14-day Relative Strength Index (RSI), a momentum indicator.
  - RSI > 70 → overbought zone, potential price correction.
  - RSI < 30 → oversold zone, potential rebound.
  - Helps assess market momentum and detect possible reversals.

---

## 5. Model Training Process: LSTM for Stock Price Prediction

### Data Preparation

- Extracted the **closing price** data from the preprocessed dataset for use as the target variable in prediction.
- Scaled the closing prices to the range **[0, 1]** using Min-Max scaling to normalize the input, which helps improve neural network training.
- Created input sequences of length **60 days** (window size), where each sample consists of the closing prices of the previous 60 days, and the corresponding target is the closing price on the next day.
- Reshaped the input data to match the expected LSTM input shape: **(samples, time steps, features)**.
- Split the dataset into **training (80%)** and **testing (20%)** sets for model validation.

### Model Architecture

- Constructed an LSTM neural network with:
  - Two LSTM layers (each with 50 units), the first returning sequences for stacking.
  - Dropout layers (20%) to reduce overfitting.
  - Dense layers for output prediction.
- The model is designed to learn temporal dependencies in stock price movements for better prediction accuracy.

### Model Training

- Compiled the model with **Adam optimizer** and **mean squared error (MSE)** loss function, appropriate for regression tasks.
- Trained the model over **50 epochs** with validation on the test set to monitor performance.

### Additional Notes

- The closing prices were inverse-transformed after training to recover predictions in original scale, allowing for meaningful comparison with actual prices.
- This approach leverages historical price patterns to forecast future closing prices, aiding investment decision-making.

---

## 6. Defining the final dataset for testing by including last 100 coloums of the Model Training with GBTRegressor 
We keep just the six core columns. ("Date","Open","High","Low","Close","Volume")
yfinance pulls historical OHLC+Volume for AAPL from Jan 1, 2020 to Jan 1, 2025.

then we Reads the CSV into a Spark DataFrame.

We create three derived features for each day 
𝑡
:

  PrevClose: yesterday’s Close (lag 1).

  MA5: 5-day moving average of Close.

  MA10: 10-day moving average of Close.

Dropping nulls removes the first 10 rows where these can’t be computed.

VectorAssembler packs all seven numeric inputs into a single features vector column.

GBTRegressor builds 100 decision-tree boosters (maxIter=100), each up to depth 5.

We chain them in a Pipeline for convenience (assembler → gbt) and train on the full engineered DataFrame.

Why GBTRegressor?
Gradient Boosting builds an ensemble of weak learners (small trees) sequentially, each one correcting its predecessor’s errors.

It often outperforms single-tree models and can naturally handle nonlinear interactions between features (e.g. how Volume and MA5 jointly influence tomorrow’s price).

The hyperparameters (maxIter, maxDepth) control complexity vs. overfitting.

and finally Rolls forward to predict the next 100 days using actual daily inputs,

Evaluates and visualizes performance vs. real market data.
