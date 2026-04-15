import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

data = {
    'StudyHours': [2, 3, 5, 1, 4, 6, 7, 8, 2, 5],
    'SleepHours': [7, 6, 8, 5, 7, 6, 7, 8, 5, 6],
    'Attendance': [60, 65, 80, 50, 75, 85, 90, 95, 55, 70],
    'Marks': [50, 55, 65, 40, 70, 80, 85, 90, 45, 68]
}

df = pd.DataFrame(data)

X=df[['StudyHours','SleepHours','Attendance']]
y=df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print("Linear Regression MSE:", mse_linear)
print("Polynomial Regression MSE:", mse_poly)
print("Random Forest Regression MSE:", mse_rf)
