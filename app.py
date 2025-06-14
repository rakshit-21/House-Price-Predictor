from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the newly trained model and categories
try:
    model = joblib.load('house_price_model_v2.pkl')
    categories = joblib.load('categories.pkl')
    print("✅ Model and categories loaded successfully!")
except Exception as e:
    print(f"❌ Loading failed: {e}")
    model = None

# Get unique values from categories
locations = list(categories[0])
green_areas = list(categories[1])
amenities = list(categories[2])
crime_rates = list(categories[3])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST" and model is not None:
        try:
            input_data = pd.DataFrame([{
                "Location": request.form["location"],
                "Size (sq ft)": float(request.form["size"]),
                "Green Area": request.form["green_area"],
                "Nearby Amenities": request.form["nearby_amenities"],
                "Crime Rate": request.form["crime_rate"]
            }])
            
            raw_prediction = model.predict(input_data)[0]
            prediction = f"₹{int(raw_prediction):,}"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        locations=locations,
        green_areas=green_areas,
        amenities=amenities,
        crime_rates=crime_rates
    )

if __name__ == "__main__":
    app.run(debug=True)