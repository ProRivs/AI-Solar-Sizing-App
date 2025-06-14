import pandas as pd
import numpy as np

# Simulate training data for components and solar kits
categories = ["batteries", "solarPanels", "inverters", "solarKits"]
components = {
    "batteries": ["mini_tubular", "tubular", "gel", "opzv", "li-ion"],
    "solarPanels": ["mono", "poly"],
    "inverters": ["mini_hybrid_pwm", "hybrid_pwm", "hybrid_mppt"],
    "solarKits": ["dc_basic_kit", "basic_kit", "standard_kit"]
}

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

for category in categories:
    if category == "solarKits":
        data = {
            "daily_energy_kwh": np.random.uniform(0.05, 0.5, n_samples),  # Kits support low loads
            "budget_xaf": np.random.uniform(50_000, 500_000, n_samples),
            "avg_cost": np.random.uniform(90_000, 290_000, n_samples),
            "lifespan": np.random.uniform(2, 7, n_samples),
            "install_type": np.random.choice([0, 1], n_samples),  # 0: rural, 1: urban
            "label": np.random.choice(components[category], n_samples)
        }
        df = pd.DataFrame(data)
        for i in range(n_samples):
            energy = df.loc[i, "daily_energy_kwh"]
            budget = df.loc[i, "budget_xaf"]
            if energy <= 0.1 and budget < 150_000:
                df.loc[i, "label"] = "dc_basic_kit"
            elif energy <= 0.15 and budget < 200_000:
                df.loc[i, "label"] = "basic_kit"
            else:
                df.loc[i, "label"] = "standard_kit"
    else:
        data = {
            "daily_energy_kwh": np.random.uniform(0.05, 20, n_samples),
            "budget_million_xaf": np.random.uniform(0.05, 1.5, n_samples),
            "avg_cost": np.random.uniform(30_000, 400_000, n_samples),
            "lifespan": np.random.choice([3, 5, 10, 15, 20], n_samples),
            "install_type": np.random.choice([0, 1], n_samples),
            "label": np.random.choice(components[category], n_samples)
        }
        df = pd.DataFrame(data)
        for i in range(n_samples):
            energy = df.loc[i, "daily_energy_kwh"]
            budget = df.loc[i, "budget_million_xaf"]
            if category == "batteries":
                if energy < 0.5 or budget < 0.2:
                    df.loc[i, "label"] = "mini_tubular"
                elif budget < 0.5:
                    df.loc[i, "label"] = "tubular"
                elif budget < 0.8:
                    df.loc[i, "label"] = "gel"
                elif budget < 1.0:
                    df.loc[i, "label"] = "opzv"
                else:
                    df.loc[i, "label"] = "li-ion"
            elif category == "solarPanels":
                df.loc[i, "label"] = "mono" if energy > 8 and budget > 0.7 else "poly"
            elif category == "inverters":
                if energy < 0.5 or budget < 0.2:
                    df.loc[i, "label"] = "mini_hybrid_pwm"
                elif energy > 10 and budget > 0.6:
                    df.loc[i, "label"] = "hybrid_mppt"
                else:
                    df.loc[i, "label"] = "hybrid_pwm"
    
    # Save to CSV
    df.to_csv(f"{category}_training_data.csv", index=False)

print("Training data generated: batteries_training_data.csv, solarPanels_training_data.csv, inverters_training_data.csv, solarKits_training_data.csv")