import json
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
from datetime import datetime, timedelta
from functools import lru_cache

def load_components():
    with open("components.json", "r") as f:
        components = json.load(f)
    # Update kit prices with real-time data
    for kit in components.get("solarKits", []):
        update_kit_price(kit)
    return components

def update_kit_price(kit):
    @lru_cache(maxsize=100)
    def scrape_price(kit_id):
        try:
            # Placeholder: Replace with actual supplier URL (e.g., https://solarctrl.cm)
            url = f"https://example.com/products/{kit_id}"
            response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            price_elem = soup.find("span", {"class": "price"})
            if price_elem:
                price = int(float(price_elem.text.replace("XAF", "").strip()))
                return {"min": price * 0.9, "avg": price, "max": price * 1.1}
            return None
        except Exception:
            return None
    price_new = scrape_price(kit["id"])
    if price_new:
        kit["price_per_system_xaf"] = price_new

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
        base_appliance = specs.get("base_appliance", load)
        power_factor = appliances[base_appliance]["power_factor"]
        real_power_w = specs["power"] * specs["qty"] * specs["hours"]
        total_energy_kwh += real_power_w / 1000
        apparent_power_va = real_power_w / power_factor
        total_apparent_power_kva += apparent_power_va / specs["hours"] / 1000
    return total_energy_kwh, total_apparent_power_kva

def predict_irradiance(location, solar_data):
    try:
        df = pd.read_csv(f"{location}_irradiance_data.csv")
        df["ds"] = pd.to_datetime(df["ds"])
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        model.add_country_holidays(country_name="CM")
        model.fit(df)
        
        today = datetime.now()
        future_dates = [today + timedelta(days=i) for i in range(7)]
        future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future)
        predicted_irradiance = forecast[["ds", "yhat"]].copy()
        predicted_irradiance["yhat"] = predicted_irradiance["yhat"].clip(lower=1.0)
        return predicted_irradiance
    except FileNotFoundError:
        avg = solar_data[location]["avg_irradiance"]
        return pd.DataFrame({
            "ds": [datetime.now() + timedelta(days=i) for i in range(7)],
            "yhat": [avg * (1 + 0.1 * np.sin(i)) for i in range(7)]
        })

def calculate_reliability_score(components_db, selected_components):
    score = 0
    max_score = 10
    for category, comp_id in selected_components.items():
        if category != "installation" and category != "solarKits":
            component = next((c for c in components_db[category] if c["id"] == comp_id), None)
            if component:
                lifespan_str = component["lifespan"].lower().replace("years", "").strip()
                if "-" in lifespan_str:
                    lifespan = float(lifespan_str.split("-")[1])
                else:
                    lifespan = float(lifespan_str)
                efficiency = 0.9 if category == "solarPanels" and comp_id == "mono" else 0.8
                score += (lifespan / 25) * 5 + efficiency * 5
    return min(score / (len(components_db) - 1) * 2, max_score) if score > 0 else 6.0

def recommend_solar_kit(daily_energy, budget, components_db, install_type, is_dc_only):
    try:
        df = pd.read_csv("solarKits_training_data.csv")
        X = df[["daily_energy_kwh", "budget_xaf", "avg_cost", "lifespan", "install_type"]]
        y = df["label"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier()
        param_grid = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_scaled, y)
        
        kits = [c for c in components_db["solarKits"] if c["is_dc_only"] == is_dc_only]
        if not kits:
            return None, "No suitable solar kit available for the specified mode."
        
        avg_costs = [kit["price_per_system_xaf"]["avg"] for kit in kits]
        lifespans = [
            float(kit["lifespan"].replace("years", "").strip().split("-")[0])
            if "-" in kit["lifespan"] else float(kit["lifespan"].replace("years", "").strip())
            for kit in kits
        ]
        
        input_features = pd.DataFrame({
            "daily_energy_kwh": [daily_energy],
            "budget_xaf": [budget],
            "avg_cost": [np.mean(avg_costs)],
            "lifespan": [np.mean(lifespans)],
            "install_type": [1 if install_type == "urban" else 0]
        })
        input_scaled = scaler.transform(input_features)
        prediction = grid_search.predict(input_scaled)[0]
        
        distances, indices = grid_search.best_estimator_.kneighbors(input_scaled)
        explanation = (
            f"Selected due to similarity with a system needing "
            f"{df.iloc[indices[0][0]]['daily_energy_kwh']:.2f} kWh/day and "
            f"budget {df.iloc[indices[0][0]]['budget_xaf']:,.0f} XAF in {install_type} setting."
        )
        
        return prediction, explanation
    except FileNotFoundError:
        kits = [c for c in components_db["solarKits"] if c["is_dc_only"] == is_dc_only]
        if not kits:
            return None, "No suitable solar kit available."
        selected = min(
            kits,
            key=lambda k: abs(daily_energy - k["supported_load_kwh"])
            if budget >= k["price_per_system_xaf"]["avg"] else float("inf")
        )
        explanation = (
            f"Selected {selected['name']} for {daily_energy:.2f} kWh/day load, "
            f"supports up to {selected['supported_load_kwh']} kWh/day."
        )
        return selected["id"], explanation

def recommend_components(
    daily_energy, budget, components_db, install_type, inverter_type,
    minimalist_mode, use_kit=False, is_dc_only=False
):
    selected_components = {}
    explanations = {}
    
    if minimalist_mode and use_kit and daily_energy <= 0.3:
        kit_id, kit_explanation = recommend_solar_kit(
            daily_energy, budget, components_db, install_type, is_dc_only
        )
        if kit_id:
            selected_components["solarKits"] = kit_id
            explanations["solarKits"] = kit_explanation
            selected_components["installation"] = "install"
            explanations["installation"] = "Installation included with solar kit."
            return selected_components, explanations
    
    categories = ["batteries", "solarPanels", "inverters"]
    for category in categories:
        if category == "inverters" and is_dc_only:
            selected_components[category] = None
            explanations[category] = "No inverter required for DC-only system."
            continue
        
        try:
            df = pd.read_csv(f"{category}_training_data.csv")
        except FileNotFoundError:
            if category == "inverters":
                selected_components[category] = "mini_hybrid_pwm" if minimalist_mode else f"hybrid_{inverter_type.lower()}"
                explanations[category] = f"Selected {'mini ' if minimalist_mode else ''}hybrid {inverter_type} inverter as per user preference{' and minimalist mode' if minimalist_mode else ''}."
            else:
                selected_components[category] = (
                    "mini_tubular" if minimalist_mode and category == "batteries" else
                    "tubular" if category == "batteries" else
                    "poly"
                )
                explanations[category] = "Default selection due to missing training data."
            continue
        
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
                "price_per_kw_xaf"
            )
            avg_costs.append(comp[price_field]["avg"])
            lifespan_str = comp["lifespan"].lower().replace("years", "").strip()
            lifespans.append(
                float(lifespan_str.split("-")[0]) if "-" in lifespan_str else float(lifespan_str)
            )
        
        input_features = pd.DataFrame({
            "daily_energy_kwh": [daily_energy],
            "budget_million_xaf": [budget / 1_000_000],
            "avg_cost": [np.mean(avg_costs)],
            "lifespan": [np.mean(lifespans)],
            "install_type": [1 if install_type == "urban" else 0]
        })
        input_scaled = scaler.transform(input_features)
        prediction = grid_search.predict(input_scaled)[0]
        
        if category == "inverters":
            prediction = "mini_hybrid_pwm" if minimalist_mode else f"hybrid_{inverter_type.lower()}"
        elif minimalist_mode or daily_energy < 0.5 or budget / 1_000_000 < 0.3:
            prediction = "mini_tubular" if category == "batteries" else "poly"
        elif budget / 1_000_000 < 0.5:
            prediction = "tubular" if category == "batteries" else "poly"
        
        distances, indices = grid_search.best_estimator_.kneighbors(input_scaled)
        explanation = (
            f"Selected due to similarity with a system needing "
            f"{df.iloc[indices[0][0]]['daily_energy_kwh']:.1f} kWh/day and "
            f"budget {df.iloc[indices[0][0]]['budget_million_xaf']*1e6:,.0f} XAF in "
            f"{install_type} setting{' with minimalist mode' if minimalist_mode else ''}."
        )
        
        selected_components[category] = prediction
        explanations[category] = explanation
    
    selected_components["installation"] = "install"
    explanations["installation"] = f"{'Minimal' if minimalist_mode or is_dc_only else 'Standard'} installation service."
    return selected_components, explanations

def generate_energy_tips(user_loads):
    tips = []
    appliances = load_appliances()
    for load, specs in user_loads.items():
        base_appliance = specs.get("base_appliance", load)
        if specs["hours"] > 12 and specs["power"] > 100 and appliances[base_appliance]["power_factor"] < 1.0:
            tips.append(f"Run {base_appliance} during peak solar hours (10 AM–2 PM) to reduce battery strain.")
        if specs["power"] > 200:
            tips.append(f"Consider energy-efficient alternatives for {base_appliance} to lower system costs.")
        if base_appliance == "Bulb" and specs["power"] > 10:
            tips.append(f"Switch to LED bulbs (5–7W) for {base_appliance} to reduce energy demand and system cost.")
    return tips if tips else ["No specific optimization tips; your usage is well-balanced."]

def get_system_suggestion(
    region, daily_energy, budget, solar_data, components_db, user_loads,
    install_type, inverter_type, minimalist_mode, use_kit=False, is_dc_only=False
):
    irradiance_forecast = predict_irradiance(region, solar_data)
    avg_irradiance = irradiance_forecast["yhat"].mean()
    
    if minimalist_mode and use_kit and daily_energy <= 0.3 and "solarKits" in components_db:
        kit_id, explanation = recommend_solar_kit(daily_energy, budget, components_db, install_type, is_dc_only)
        if kit_id:
            kit = next((k for k in components_db["solarKits"] if k["id"] == kit_id), None)
            if kit:
                return {
                    "battery_size": kit["supported_load_kwh"] * 2,
                    "production": kit["supported_load_kwh"] * 1.2,
                    "budget": kit["price_per_system_xaf"]["avg"],
                    "components": [kit["name"], components_db["installation"]["name"]],
                    "irradiance_forecast": irradiance_forecast,
                    "production_forecast": irradiance_forecast.copy(),
                    "explanations": {"solarKits": explanation},
                    "inverter_capacity_kw": 0.0 if kit["is_dc_only"] else 0.3,
                    "reliability_score": 6.0
                }
    
    panel_capacity_kw = (daily_energy / avg_irradiance) / 0.85 * 1.2
    battery_capacity_kwh = max(daily_energy * (1.5 if minimalist_mode else 2) / 0.5, 0.5 if minimalist_mode else 1.0)
    _, total_apparent_power_kva = calculate_energy_demand(user_loads)
    inverter_capacity_kw = (0.0 if is_dc_only else
        max(total_apparent_power_kva, daily_energy / 24 * (1.1 if minimalist_mode else 1.2), 0.3 if minimalist_mode else 0.5)
    )
    
    selected_components, explanations = recommend_components(
        daily_energy, budget, components_db, install_type, inverter_type, minimalist_mode, use_kit, is_dc_only
    )
    
    total_cost = {"min": 0, "max": 0, "avg": 0}
    components = components_db
    for category, comp_id in selected_components.items():
        if category == "installation":
            component = components["installation"]
            for key in ["min", "max", "avg"]:
                total_cost[key] += component["price_per_system_xaf"][key] * (
                    0.5 if minimalist_mode or is_dc_only else 1.0
                )
        elif category == "solarKits" and comp_id:
            component = next((c for c in components["solarKits"] if c["id"] == comp_id), None)
            if component:
                for key in ["min", "max", "avg"]:
                    total_cost[key] += component["price_per_system_xaf"][key]
        elif comp_id:
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
    
    production_forecast = irradiance_forecast.copy()
    production_forecast["yhat"] = production_forecast["yhat"] * panel_capacity_kw * 0.85
    
    return {
        "battery_size": round(battery_capacity_kwh, 2),
        "production": round(panel_capacity_kw * avg_irradiance * 0.85, 2),
        "budget": round(total_cost["avg"]),
        "components": [
            components[cat]["name"] if cat == "installation" else
            next(c["name"] for c in components[cat] if c["id"] == comp_id)
            for cat, comp_id in selected_components.items() if comp_id
        ],
        "irradiance_forecast": irradiance_forecast,
        "production_forecast": production_forecast,
        "explanations": explanations,
        "inverter_capacity_kw": round(inverter_capacity_kw, 2),
        "reliability_score": calculate_reliability_score(components_db, selected_components)
    }

def calculate_cost(components, selections, is_dc_only=False):
    total_cost = {"min": 0, "max": 0, "avg": 0}
    component_costs = {"battery": 0, "panel": 0, "inverter": 0, "installation": 0, "kit": 0}
    
    if "solarKits" in selections and selections["solarKits"]:
        kit = next((k for k in components["solarKits"] if k["id"] == selections["solarKits"]), None)
        if kit:
            for key in ["min", "max", "avg"]:
                total_cost[key] += kit["price_per_system_xaf"][key]
            component_costs["kit"] = round(kit["price_per_system_xaf"]["avg"])
            component_costs["installation"] = 0
            return {"components": component_costs, **{k: round(v) for k, v in total_cost.items()}}
    
    battery = next((b for b in components["batteries"] if b["id"] == selections["battery"]), None)
    if battery:
        for key in ["min", "max", "avg"]:
            cost = battery["price_per_kwh_xaf"][key] * selections["battery_capacity_kwh"]
            total_cost[key] += cost
            if key == "avg":
                component_costs["battery"] = round(cost)
    panel = next((p for p in components["solarPanels"] if p["id"] == selections["panel"]), None)
    if panel:
        for key in ["min", "max", "avg"]:
            cost = panel["price_per_watt_xaf"][key] * selections["panel_capacity_kw"] * 1000
            total_cost[key] += cost
            if key == "avg":
                component_costs["panel"] = round(cost)
    if not is_dc_only:
        inverter = next((i for i in components["inverters"] if i["id"] == selections["inverter"]), None)
        if inverter:
            for key in ["min", "max", "avg"]:
                cost = inverter["price_per_kw_xaf"][key] * selections["inverter_capacity_kw"]
                total_cost[key] += cost
                if key == "avg":
                    component_costs["inverter"] = round(cost)
    installation = components["installation"]
    for key in ["min", "max", "avg"]:
        cost = installation["price_per_system_xaf"][key] * (
            0.5 if selections.get("battery") == "mini_tubular" or is_dc_only else 1.0
        )
        total_cost[key] += cost
        if key == "avg":
            component_costs["installation"] = round(cost)
    return {"components": component_costs, **{k: round(v) for k, v in total_cost.items()}}