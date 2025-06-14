import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset
data = pd.read_csv('delhi_house_prices-4.csv')  # Save your CSV data with this name

# Preprocessing
categorical_features = ['Location', 'Green Area', 'Nearby Amenities', 'Crime Rate']
numeric_features = ['Size (sq ft)']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200, 
        random_state=42,
        max_depth=10,
        min_samples_leaf=2
    ))
])

# Split data
X = data.drop('Price (INR)', axis=1)
y = data['Price (INR)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")

# Save the retrained model
joblib.dump(model, 'house_price_model_v2.pkl')
print("Model retrained and saved successfully!")

# Save the categories for the Flask app
encoder = model.named_steps['preprocessor'].named_transformers_['cat']
joblib.dump(encoder.categories_, 'categories.pkl')