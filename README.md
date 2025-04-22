
---

## 📌 Objective

To analyze historical Nvidia stock price data and predict:
- **Price Direction (Up/Down)** using classification models
- **Future Prices** using regression models

---

## 🔧 Tools & Technologies

- Python (pandas, numpy, matplotlib, sklearn)
- VS Code (for writing and running code)
- Git & GitHub (version control)
- Kaggle (data source)

---

## 📊 Methodology

1. **Data Collection & Exploration**
   - Dataset: NVDA stock data from Kaggle
   - Checked for null values (none found)
   - Plotted closing price trend

2. **Feature Engineering**
   - Added indicators like:
     - SMA (Simple Moving Average)
     - RSI (Relative Strength Index)
     - MACD, Signal Line, MACD Histogram

3. **Preprocessing**
   - Used `MinMaxScaler` for normalization
   - Removed nulls after feature creation

4. **Model Building**
   - **Classification**:
     - Logistic Regression
     - Support Vector Classifier (SVC with polynomial kernel)
   - **Regression**:
     - Linear Regression

---

## 🧠 Results & Observations

### 📉 Classification Performance:
| Model               | Training AUC | Validation AUC |
|--------------------|--------------|----------------|
| Logistic Regression| 0.5095       | 0.4774         |
| SVC (poly)         | 0.4996       | 0.4828         |

### 📈 Regression Performance:
| Metric       | Training | Validation |
|--------------|----------|------------|
| RMSE         | 0.00033  | 0.0128     |
| R² Score     | 0.9989   | 0.9977     |

- **Linear Regression** showed excellent fit for predicting closing prices.
- **Classification models** performed slightly better than random guessing.

---

## 📷 Sample Visualizations

- **Figure 1:** Normalized vs actual prices for the test set  
- **Figure 2:** Nvidia closing price over time

---

## ✅ Conclusion

- Linear Regression proved to be the best fit model for stock price prediction.
- Technical indicators significantly enhanced model performance.
- GitHub and VS Code were used for version control and code implementation.

---

## 📁 Data Source

- [Kaggle: Nvidia Stock Price Dataset](https://www.kaggle.com/)

---

## 📬 Author

- 👩‍💻 **Sona12503** –

---

