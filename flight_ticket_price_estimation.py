import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("flight_data.csv")
print(data.head())
print("\nShape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

X = data.drop(columns=["flight_id","ticket_price"])
y = data["ticket_price"]

num = ["travel_month","days_before_booking","duration_hours"]
cat = ["airline","source","destination","stops"]

pre = ColumnTransformer([
    ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
    ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
model = Pipeline([("preprocessor",pre),("regressor",LinearRegression())])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**.5
r2=r2_score(y_test,y_pred)
print(f"\nMAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

comparison=pd.DataFrame({"Actual Ticket Price":y_test.values,"Predicted Ticket Price":y_pred.round(2)})
print("\nActual vs Predicted:")
print(comparison.head(10))

new_flight=pd.DataFrame([{"airline":"Vistara","source":"Delhi","destination":"Mumbai","travel_month":12,"days_before_booking":30,"stops":"Non-stop","duration_hours":2.2}])
print(f"\nEstimated Flight Ticket Price: ₹{model.predict(new_flight)[0]:,.2f}")

names=model.named_steps["preprocessor"].get_feature_names_out()
coefs=model.named_steps["regressor"].coef_
importance=pd.DataFrame({"feature":names,"coefficient":coefs,"absolute_impact":abs(coefs)}).sort_values("absolute_impact",ascending=False)
print("\nTop Factors:")
print(importance.head(15)[["feature","coefficient"]])

plt.figure(figsize=(8,5)); plt.scatter(data["duration_hours"],data["ticket_price"],alpha=.6)
plt.xlabel("Flight Duration (hours)"); plt.ylabel("Ticket Price"); plt.title("Flight Duration vs Ticket Price"); plt.tight_layout(); plt.savefig("duration_vs_price.png",dpi=150); plt.show()

plt.figure(figsize=(8,5)); plt.scatter(data["days_before_booking"],data["ticket_price"],alpha=.6)
plt.xlabel("Days Before Booking"); plt.ylabel("Ticket Price"); plt.title("Booking Lead Time vs Ticket Price"); plt.tight_layout(); plt.savefig("booking_days_vs_price.png",dpi=150); plt.show()

data.groupby("airline",dropna=False)["ticket_price"].mean().sort_values().plot(kind="bar",figsize=(9,5))
plt.xlabel("Airline"); plt.ylabel("Average Ticket Price"); plt.title("Average Ticket Price by Airline"); plt.xticks(rotation=30,ha="right"); plt.tight_layout(); plt.savefig("airline_vs_price.png",dpi=150); plt.show()

plt.figure(figsize=(8,5)); plt.scatter(y_test,y_pred,alpha=.6)
plt.xlabel("Actual Ticket Price"); plt.ylabel("Predicted Ticket Price"); plt.title("Actual vs Predicted Flight Ticket Prices"); plt.tight_layout(); plt.savefig("actual_vs_predicted.png",dpi=150); plt.show()
