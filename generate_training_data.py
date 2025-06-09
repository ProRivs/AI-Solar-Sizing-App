import pandas as pd
import numpy as np

# Simulate training data for each component category
categories = ["batteries", "solarPanels", "inverters", "chargeControllers"]
components = {
    "batteries": ["li-ion", "tubular"],
    "solarPanels": ["mono", "poly"],
    "inverters": ["string", "micro"],
    "chargeControllers": ["pwm", "mppt"]
}

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

for category in categories:
    data = {
        "daily_energy_kwh": np.random.uniform(1, 20, n_samples),  # Daily energy demand in kWh
        "budget_million_xaf": np.random.uniform(0.1, 1.5, n_samples),  # Budget in million XAF
        "avg_cost": np.random.uniform(100_000, 500_000, n_samples),  # Average component cost in XAF
        "lifespan": np.random.choice([5, 10, 15, 20], n_samples),  # Component lifespan in years
        "install_type": np.random.choice([0, 1], n_samples),  # 0: rural, 1: urban
        "label": np.random.choice(components[category], n_samples)  # Component ID
    }
    
    # Adjust labels based on heuristic rules
    df = pd.DataFrame(data)
    for i in range(n_samples):
        energy = df.loc[i, "daily_energy_kwh"]
        budget = df.loc[i, "budget_million_xaf"]
        if category == "batteries":
            df.loc[i, "label"] = "li-ion" if energy > 10 and budget > 0.5 else "tubular"
        elif category == "solarPanels":
            df.loc[i, "label"] = "mono" if energy > 8 and budget > 0.7 else "poly"
        elif category == "inverters":
            df.loc[i, "label"] = "micro" if energy > 12 and budget > 0.8 else "string"
        elif category == "chargeControllers":
            df.loc[i, "label"] = "mppt" if energy > 5 and budget > 0.4 else "pwm"
    
    # Save to CSV
    df.to_csv(f"{category}_training_data.csv", index=False)

print("Training data generated: batteries_training_data.csv, solarPanels_training_data.csv, inverters_training_data.csv, chargeControllers_training_data.csv")