import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
from datetime import datetime, timedelta

def load_components():
    with open("components.json", "r") as f:
        return json.load(f)

def load_solar_irradiance():
    with open("solar_irradiance.json", "r") as f:
        return json.load(f)

def load_appliances():
    with open("appliances.json", "r") as f:
        return json.load(f)

def calculate_energy_demand(user_loads):
    total_energy_kwh = 0
    total_apparent_power_kva = 0
    appliances = load_appliances()
    for load, specs in user_loads.items():
        base_appliance = specs.get("base_appliance", load)  # Use base_appliance or fallback to load
        power_factor = appliances[base_appliance]["power_factor"]
        real_power_w = specs["power"] * specs["qty"] * specs["hours"]
        total_energy_kwh += real_power_w / 1000  # Convert Wh to kWh
        apparent_power_va = real_power_w / power_factor
        total_apparent_power_kva += apparent_power_va / specs["hours"] / 1000  # kVA for inverter sizing
    return total_energy_kwh, total_apparent_power_kva

def predict_irradiance(location, solar_data):
    try:
        df = pd.read_csv(f"{location}_irradiance_data.csv")
        df["ds"] = pd.to_datetime(df["ds"])
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        
        # Predict for next 7 days
        today = datetime.now()
        future_dates = [today + timedelta(days=i) for i in range(7)]
        future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future)
        predicted_irradiance = forecast[["ds", "yhat"]].copy()
        predicted_irradiance["yhat"] = predicted_irradiance["yhat"].clip(lower=1.0)
        return predicted_irradiance  # DataFrame with ds (dates) and yhat (kWh/m²/day)
    except FileNotFoundError:
        avg = solar_data[location]["avg_irradiance"]
        return pd.DataFrame({
            "ds": [datetime.now() + timedelta(days=i) for i in range(7)],
            "yhat": [avg] * 7
        })

def recommend_components(daily_energy, budget, components_db, install_type="urban"):
    selected_components = {}
    explanations = {}
    categories = ["batteries", "solarPanels", "inverters", "chargeControllers"]
    
    for category in categories:
        try:
            df = pd.read_csv(f"{category}_training_data.csv")
        except FileNotFoundError:
            selected_components[category] = (
                "tubular" if category == "batteries" else
                "poly" if category == "solarPanels" else
                "string" if category == "inverters" else
                "mppt"
            )
            explanations[category] = "Default selection due to missing training data."
            continue
        
        # Ensure correct column names
        X = df[["daily_energy_kwh", "budget_million_xaf", "avg_cost", "lifespan", "install_type"]]
        y = df["label"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier()
        param_grid = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_scaled, y)
        
        avg_costs = []
        lifespans = []
        for comp in components_db[category]:
            price_field = (
                "price_per_kwh_xaf" if category == "batteries" else
                "price_per_watt_xaf" if category == "solarPanels" else
                "price_per_kw_xaf" if category == "inverters" else
                "price_per_unit_xaf"
            )
            avg_costs.append(comp[price_field]["avg"])
            lifespans.append(float(comp["lifespan"].split("-")[0]) if "-" in comp["lifespan"] else float(comp["lifespan"]))
        
        # Create input features as DataFrame to preserve column names
        input_features = pd.DataFrame({
            "daily_energy_kwh": [daily_energy],
            "budget_million_xaf": [budget / 1_000_000],
            "avg_cost": [np.mean(avg_costs)],
            "lifespan": [np.mean(lifespans)],
            "install_type": [1 if install_type == "urban" else 0]
        })
        input_scaled = scaler.transform(input_features)
        prediction = grid_search.predict(input_scaled)[0]
        distances, indices = grid_search.best_estimator_.kneighbors(input_scaled)
        explanation = f"Selected due to similarity with a system needing {df.iloc[indices[0][0]]['daily_energy_kwh']:.1f} kWh/day and budget {df.iloc[indices[0][0]]['budget_million_xaf']*1e6:,.0f} XAF in {install_type} setting."
        
        selected_components[category] = prediction
        explanations[category] = explanation
    
    selected_components["installation"] = "install"
    explanations["installation"] = "Standard installation service."
    return selected_components, explanations

def generate_energy_tips(user_loads):
    tips = []
    appliances = load_appliances()
    for load, specs in user_loads.items():
        base_appliance = specs.get("base_appliance", load)
        if specs["hours"] > 12 and specs["power"] > 100 and appliances[base_appliance]["power_factor"] < 1.0:
            tips.append(f"Run {base_appliance} during peak solar hours (10 AM–2 PM) to reduce battery and inverter strain.")
    return tips if tips else ["No specific optimization tips; your usage is well-balanced."]

def get_system_suggestion(region, daily_energy, budget, solar_data, components_db, user_loads, install_type="urban"):
    irradiance_forecast = predict_irradiance(region, solar_data)
    avg_irradiance = irradiance_forecast["yhat"].mean()
    panel_capacity_kw = (daily_energy / avg_irradiance) / 0.8 * 1.2
    battery_capacity_kwh = daily_energy * 2 / 0.5
    _, total_apparent_power_kva = calculate_energy_demand(user_loads)
    inverter_capacity_kw = max(total_apparent_power_kva, daily_energy / 24 * 1.2)
    
    selected_components, explanations = recommend_components(daily_energy, budget, components_db, install_type)
    
    total_cost = {"min": 0, "max": 0, "avg": 0}
    components = components_db
    for category, comp_id in selected_components.items():
        if category != "installation":
            component = next((c for c in components[category] if c["id"] == comp_id), None)
            if component:
                if category == "batteries":
                    for key in ["min", "max", "avg"]:
                        total_cost[key] += component["price_per_kwh_xaf"][key] * battery_capacity_kwh
                elif category == "solarPanels":
                    for key in ["min", "max", "avg"]:
                        total_cost[key] += component["price_per_watt_xaf"][key] * panel_capacity_kw * 1000
                elif category == "inverters":
                    for key in ["min", "max", "avg"]:
                        total_cost[key] += component["price_per_kw_xaf"][key] * inverter_capacity_kw
                else:
                    for key in ["min", "max", "avg"]:
                        total_cost[key] += component["price_per_unit_xaf"][key]
        else:
            for key in ["min", "max", "avg"]:
                total_cost[key] += components[category]["price_per_system_xaf"][key]
    
    # Calculate 7-day energy production
    production_forecast = irradiance_forecast.copy()
    production_forecast["yhat"] = production_forecast["yhat"] * panel_capacity_kw * 0.8  # kWh/day
    
    return {
        "battery_size": round(battery_capacity_kwh, 2),
        "production": round(panel_capacity_kw * avg_irradiance * 0.8, 2),
        "budget": round(total_cost["avg"]),
        "components": [components[cat]["name"] if cat == "installation" else next(c["name"] for c in components[cat] if c["id"] == comp_id) for cat, comp_id in selected_components.items()],
        "irradiance_forecast": irradiance_forecast,
        "production_forecast": production_forecast,
        "explanations": explanations,
        "inverter_capacity_kw": round(inverter_capacity_kw, 2)
    }

def calculate_cost(components, selections):
    total_cost = {"min": 0, "max": 0, "avg": 0}
    battery = next((b for b in components["batteries"] if b["id"] == selections["battery"]), None)
    if battery:
        for key in ["min", "max", "avg"]:
            total_cost[key] += battery["price_per_kwh_xaf"][key] * selections["battery_capacity_kwh"]
    panel = next((p for p in components["solarPanels"] if p["id"] == selections["panel"]), None)
    if panel:
        for key in ["min", "max", "avg"]:
            total_cost[key] += panel["price_per_watt_xaf"][key] * selections["panel_capacity_kw"] * 1000
    inverter = next((i for i in components["inverters"] if i["id"] == selections["inverter"]), None)
    if inverter:
        for key in ["min", "max", "avg"]:
            total_cost[key] += inverter["price_per_kw_xaf"][key] * selections["inverter_capacity_kw"]
    controller = next((c for c in components["chargeControllers"] if c["id"] == selections["charge_controller"]), None)
    if controller:
        for key in ["min", "max", "avg"]:
            total_cost[key] += controller["price_per_unit_xaf"][key]
    installation = components["installation"]
    for key in ["min", "max", "avg"]:
        total_cost[key] += installation["price_per_system_xaf"][key]
    return {k: round(v) for k, v in total_cost.items()}