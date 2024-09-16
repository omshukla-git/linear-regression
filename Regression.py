import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and Prepare Data
data = {'Feature': [1, 2, 3, 4, 5], 'Target': [2, 4, 5, 4, 5]}
df = pd.DataFrame(data)

# Separate feature (X) and target (y)
X = df[['Feature']]
y = df['Target']

# Step 2: Train the Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 4: Visualize the Regression Line
plt.scatter(X, y, color='blue')  # Scatter plot of data
plt.plot(X, model.predict(X), color='red')  # Regression line
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.show()
