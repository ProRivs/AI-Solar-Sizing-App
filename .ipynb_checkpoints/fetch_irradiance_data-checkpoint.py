import requests
import pandas as pd
from datetime import datetime, timedelta

# NASA POWER API endpoint
base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Coordinates for Yaounde, Douala, Garoua
locations = {
    "Yaounde": {"lat": 3.848, "lon": 11.502},
    "Douala": {"lat": 4.051, "lon": 9.704},
    "Garoua": {"lat": 9.301, "lon": 13.398}
}

# Parameters
parameters = "ALLSKY_SFC_SW_DWN"  # Surface solar irradiance (kWh/m²/day)
start_date = "20150101"  # Extended to 2015 for robust training
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")  # Up to yesterday

# Fetch data
data = {}
for city, coords in locations.items():
    url = f"{base_url}?parameters={parameters}&community=RE&longitude={coords['lon']}&latitude={coords['lat']}&start={start_date}&end={end_date}&format=JSON"
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        dates = []
        irradiance = []
        for date, value in json_data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].items():
            if value != -999:
                dates.append(datetime.strptime(date, "%Y%m%d"))
                irradiance.append(value)
        data[city] = pd.DataFrame({"ds": dates, "y": irradiance})
    else:
        print(f"Failed to fetch data for {city}")

# Save data
for city, df in data.items():
    df.to_csv(f"{city}_irradiance_data.csv", index=False)

print("Irradiance data saved: Yaounde_irradiance_data.csv, Douala_irradiance_data.csv, Garoua_irradiance_data.csv")