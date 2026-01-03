import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# Make sure 'parkinsons.csv' is in the same folder as this script
df = pd.read_csv('parkinsons.csv')

# 2. Select features
# Inputs: 'PPE' and 'spread1' are the most important voice measures
# Output: 'status' (0 = healthy, 1 = Parkinson's)
X = df[['PPE', 'spread1']]
y = df['status']

# 3. Scale the data (Makes all numbers between 0 and 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Choose and Train the model
# RandomForest is very beginner-friendly and accurate
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Test the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Current Accuracy: {accuracy}")

# 7. Save the model if accuracy is 0.8 or higher
if accuracy >= 0.8:
    joblib.dump(model, 'parkinsons_model.joblib')
    print("Success! Model saved as parkinsons_model.joblib")
else:
    print("Accuracy too low. Try adding more features.")
