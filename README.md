# Macro Sensitivity Regression Model

This project models the relationship between stock returns and macroeconomic factors such as CPI, Fed Funds Rate, Oil, Unemployment, and VIX.

It is part of a broader research effort to understand portfolio performance during volatile periods such as the current trade war and tariffs.

---

## Files

- **`macro_sensitivity_model.py`**  
  Python script to run a linear regression between stock returns and macroeconomic variables.

- **`FRED_macro_data_2000_2025.csv`**  
  Historical macroeconomic data (2000–2025) were used as input for the model.

---

## How to Use

1. Install necessary Python libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib finance
2. Run the main model:
      ```bash
   python macro_sensitivity_model.py
