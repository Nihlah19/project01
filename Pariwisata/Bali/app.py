from flask import Flask, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__,     static_folder='../static'
)

@app.route('/')
def index():
    data = pd.read_csv('bali.csv')
    data = data.dropna(subset=['lokasi', 'name', 'rating']).reset_index(drop=True)
    data['rating'] = data['rating'].astype(float)

    # Urutkan berdasarkan rating asli
    top_data = data.sort_values(by='rating', ascending=False).head(3)

    le_name = LabelEncoder()
    le_lokasi = LabelEncoder()
    top_data['name_encoded'] = le_name.fit_transform(top_data['name'])
    top_data['lokasi_encoded'] = le_lokasi.fit_transform(top_data['lokasi'])

    X = top_data[['lokasi_encoded', 'name_encoded']]
    y = top_data['rating']

    model = RandomForestRegressor()
    model.fit(X, y)
    pred = model.predict(X)

    results = []
    for i in range(len(top_data)):
        rating = round(top_data['rating'].iloc[i], 2)
        full_stars = int(rating)
        empty_stars = 5 - full_stars
        stars_html = '★' * full_stars + '☆' * empty_stars

        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={top_data['name'].iloc[i].replace(' ', '+')}"
        results.append({
            "name": top_data['name'].iloc[i],
            "rating_asli": round(top_data['rating'].iloc[i], 2),
            "rating_prediksi": round(pred[i], 2),
            "google_maps_url": google_maps_url,
            "stars": stars_html
        })

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050)
