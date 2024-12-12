from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

xls=pd.ExcelFile('Coffee Shop Sales.xlsx') # there were spaces in the filename :)
df = pd.read_excel(xls, 'Transactions')
print(df)


# 2. Data Preprocessing
# Convert 'Revenue' to numeric (removing $ sign if present)
df['Revenue'] = df['Revenue'].replace('[\$,]', '', regex=True).astype(float)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, columns=[
                    'store_location', 'product_category', 'product_type'], drop_first=True)

# Inspect the data to ensure it's ready
print(df.head())
print(df.dtypes)

# 3. Define the Dependent and Independent Variables
# Example: Using 'transaction_qty', 'unit_price', 'store_id' as predictors
# Add more features as needed
X = df[['transaction_qty', 'unit_price', 'store_id']]
Y = df['Revenue']

# 4. Split the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# 5. Fit the Model
model = LinearRegression()
model.fit(X_train, Y_train)

# 6. Evaluate the Model
Y_pred = model.predict(X_test)

# Calculate and print metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 7. Interpret the Coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# 8. Make Predictions
# Example: Predicting Revenue for a new transaction with given features
# Example input for prediction [transaction_qty, unit_price, store_id]
new_data = [[2, 3.5, 5]]
predicted_revenue = model.predict(new_data)
print(f"Predicted Revenue: {predicted_revenue[0]}")