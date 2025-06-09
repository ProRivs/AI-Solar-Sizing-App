import json

components_data = {
    "batteries": [
        {
            "id": "li-ion",
            "name": "Lithium-Ion Battery",
            "tagline": "High efficiency, long lifespan",
            "description": "Advanced lithium-ion battery for reliable energy storage.",
            "best_for": "Urban households, high-budget systems",
            "lifespan": "10-15 years",
            "price_per_kwh_xaf": {"min": 400_000, "avg": 450_000, "max": 500_000},
            "pros": ["Long lifespan", "High efficiency"],
            "cons": ["High cost"]
        },
        {
            "id": "tubular",
            "name": "Tubular Battery",
            "tagline": "Affordable and durable",
            "description": "Lead-acid tubular battery, cost-effective for rural use.",
            "best_for": "Rural households, low-budget systems",
            "lifespan": "5-8 years",
            "price_per_kwh_xaf": {"min": 200_000, "avg": 250_000, "max": 300_000},
            "pros": ["Low cost", "Durable"],
            "cons": ["Shorter lifespan"]
        }
    ],
    "solarPanels": [
        {
            "id": "mono",
            "name": "Monocrystalline Panel",
            "tagline": "High efficiency, sleek design",
            "description": "High-efficiency panels for maximum power output.",
            "best_for": "Space-constrained roofs",
            "lifespan": "20-25 years",
            "price_per_watt_xaf": {"min": 600, "avg": 700, "max": 800},
            "pros": ["High efficiency", "Long lifespan"],
            "cons": ["Expensive"]
        },
        {
            "id": "poly",
            "name": "Polycrystalline Panel",
            "tagline": "Cost-effective solar power",
            "description": "Affordable panels with decent efficiency.",
            "best_for": "Large installations, budget systems",
            "lifespan": "15-20 years",
            "price_per_watt_xaf": {"min": 400, "avg": 500, "max": 600},
            "pros": ["Affordable", "Good performance"],
            "cons": ["Lower efficiency"]
        }
    ],
    "inverters": [
        {
            "id": "string",
            "name": "String Inverter",
            "tagline": "Reliable and cost-effective",
            "description": "Central inverter for simple solar systems.",
            "best_for": "Standard installations",
            "lifespan": "10-15 years",
            "price_per_kw_xaf": {"min": 300_000, "avg": 350_000, "max": 400_000},
            "pros": ["Cost-effective", "Reliable"],
            "cons": ["Less flexible"]
        },
        {
            "id": "micro",
            "name": "Micro Inverter",
            "tagline": "Optimized for efficiency",
            "description": "Per-panel inverters for maximum output.",
            "best_for": "Complex roofs, high-budget systems",
            "lifespan": "15-20 years",
            "price_per_kw_xaf": {"min": 500_000, "avg": 600_000, "max": 700_000},
            "pros": ["High efficiency", "Flexible"],
            "cons": ["Expensive"]
        }
    ],
    "chargeControllers": [
        {
            "id": "pwm",
            "name": "PWM Charge Controller",
            "tagline": "Simple and affordable",
            "description": "Pulse-width modulation controller for basic systems.",
            "best_for": "Small, budget systems",
            "lifespan": "5-10 years",
            "price_per_unit_xaf": {"min": 50_000, "avg": 75_000, "max": 100_000},
            "pros": ["Low cost", "Simple"],
            "cons": ["Lower efficiency"]
        },
        {
            "id": "mppt",
            "name": "MPPT Charge Controller",
            "tagline": "Maximum power point tracking",
            "description": "Advanced controller for optimized charging.",
            "best_for": "Large, high-efficiency systems",
            "lifespan": "10-15 years",
            "price_per_unit_xaf": {"min": 150_000, "avg": 200_000, "max": 250_000},
            "pros": ["High efficiency", "Versatile"],
            "cons": ["Higher cost"]
        }
    ],
    "installation": {
        "id": "install",
        "name": "Installation",
        "tagline": "Professional setup",
        "description": "Complete installation service for solar systems.",
        "best_for": "All systems",
        "lifespan": "N/A",
        "price_per_system_xaf": {"min": 100_000, "avg": 150_000, "max": 200_000},
        "pros": ["Professional", "Reliable"],
        "cons": ["Additional cost"]
    }
}

solar_irradiance_data = {
    "Yaounde": {"avg_irradiance": 5.0, "min_irradiance": 4.5, "max_irradiance": 5.5},
    "Douala": {"avg_irradiance": 4.8, "min_irradiance": 4.3, "max_irradiance": 5.3},
    "Garoua": {"avg_irradiance": 5.2, "min_irradiance": 4.7, "max_irradiance": 5.7}
}

appliances_data = {
    "Bulb": {"power": 10, "typical_hours": 5.0, "power_factor": 1.0},
    "Fan": {"power": 50, "typical_hours": 8.0, "power_factor": 0.8},
    "TV": {"power": 100, "typical_hours": 4.0, "power_factor": 0.9},
    "Refrigerator": {"power": 150, "typical_hours": 24.0, "power_factor": 0.85},
    "Phone Charger": {"power": 5, "typical_hours": 2.0, "power_factor": 1.0}
}

with open("components.json", "w") as f:
    json.dump(components_data, f, indent=2)

with open("solar_irradiance.json", "w") as f:
    json.dump(solar_irradiance_data, f, indent=2)

with open("appliances.json", "w") as f:
    json.dump(appliances_data, f, indent=2)

print("Data files generated: components.json, solar_irradiance.json, appliances.json")