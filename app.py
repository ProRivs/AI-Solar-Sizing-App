import streamlit as st
import plotly.graph_objects as go
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
from logic import calculate_energy_demand, get_system_suggestion, load_solar_irradiance, load_appliances, load_components, generate_energy_tips, calculate_cost

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("PDF export feature is unavailable because the 'reportlab' package is not installed. Install it using 'pip install reportlab'.")

st.set_page_config(page_title="Solar Sizing Tool", layout="centered")

# Load Data Files
with open("solar_irradiance.json", "r") as f:
    solar_data = json.load(f)

with open("components.json", "r") as f:
    components_db = json.load(f)

with open("appliances.json", "r") as f:
    load_prices = json.load(f)

# Title & Intro
st.title("☀️ Solar Sizing Tool (Cameroon)")
st.markdown("Optimize your solar system sizing with AI-driven recommendations.")
st.markdown("---")

# Inputs
st.markdown("## 🗺️ Location & Budget Input")
col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("📍 Select Your Region", list(solar_data.keys()))
    install_type = st.selectbox("🏠 Installation Type", ["urban", "rural"])

with col2:
    upper_budget = st.slider("💰 Max Budget (XAF)", min_value=100_000, max_value=1_500_000, step=50_000)

st.markdown("## ⚡ Appliance Load Estimation")
st.markdown("Select appliances and specify power, hours, and quantity:")
selected_appliances = st.multiselect("📋 Select Appliances", list(load_prices.keys()))
user_loads = {}
if selected_appliances:
    for appliance in selected_appliances:
        num_instances = st.number_input(
            f"Number of {appliance} Instances",
            min_value=0, max_value=10, value=1, step=1,
            key=f"num_instances_{appliance}"
        )
        for i in range(num_instances):
            st.subheader(f"{appliance} (Instance {i+1})")
            power = st.slider(
                f"{appliance} Power (W)",
                min_value=5, max_value=500, value=load_prices[appliance]["power"], step=5,
                key=f"power_{appliance}_{i}"
            )
            hours = st.slider(
                f"{appliance} Hours per Day",
                min_value=0.0, max_value=24.0, value=float(load_prices[appliance]["typical_hours"]), step=0.1,
                key=f"hours_{appliance}_{i}"
            )
            qty = st.number_input(
                f"{appliance} Quantity",
                min_value=0, max_value=20, value=1, step=1,
                key=f"qty_{appliance}_{i}"
            )
            st.write(f"**Power Factor**: {load_prices[appliance]['power_factor']}")
            if qty > 0 and hours > 0:
                user_loads[f"{appliance}_{i}"] = {
                    "power": power,
                    "qty": qty,
                    "hours": hours,
                    "base_appliance": appliance
                }

st.markdown("---")

# Calculation & Output
if user_loads:
    # Energy demand
    daily_energy, apparent_power_kva = calculate_energy_demand(user_loads)
    st.success(f"🔋 Daily Demand: **{daily_energy:.2f} kWh/day**")
    st.write(f"**Apparent Power**: {apparent_power_kva:.2f} kVA (for inverter sizing)")
    
    # Solar system suggestion
    suggestion = get_system_suggestion(region, daily_energy, upper_budget, solar_data, components_db, user_loads, install_type)
    
    # Display metrics
    st.markdown("## 🔍 System Recommendation")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔋 Battery Size", f"{suggestion['battery_size']} kWh")
    c2.metric("☀️ Avg Daily Output", f"{suggestion['production']} kWh")
    c3.metric("💵 Total System Cost", f"{suggestion['budget']:,} XAF")
    
    # Irradiance and Production Forecasts
    st.markdown("### 📈 7-Day Forecasts")
    irradiance_fig = go.Figure()
    irradiance_fig.add_trace(go.Scatter(
        x=suggestion["irradiance_forecast"]["ds"],
        y=suggestion["irradiance_forecast"]["yhat"],
        mode="lines+markers",
        name="Irradiance",
        line=dict(color="blue")
    ))
    irradiance_fig.update_layout(
        title="Solar Irradiance Forecast (7 Days)",
        xaxis_title="Date",
        yaxis_title="kWh/m²/day",
        template="simple_white"
    )
    st.plotly_chart(irradiance_fig)

    production_fig = go.Figure()
    production_fig.add_trace(go.Scatter(
        x=suggestion["production_forecast"]["ds"],
        y=suggestion["production_forecast"]["yhat"],
        mode="lines+markers",
        name="Production",
        line=dict(color="orange")
    ))
    production_fig.update_layout(
        title="Energy Production Forecast (7 Days)",
        xaxis_title="Date",
        yaxis_title="kWh/day",
        template="simple_white"
    )
    st.plotly_chart(production_fig)

    # Component Override
    st.markdown("### 🧰 Component Selection")
    selected_components = {}
    for category in ["batteries", "solarPanels", "inverters", "chargeControllers"]:
        options = [comp["name"] for comp in components_db[category]]
        default = next(c["name"] for c in components_db[category] if c["name"] in suggestion["components"])
        selected = st.selectbox(f"Select {category.title()}", options, index=options.index(default), key=f"select_{category}")
        selected_components[category] = next(c["id"] for c in components_db[category] if c["name"] == selected)
        st.write(f"**AI Explanation**: {suggestion['explanations'][category]}")
    selected_components["installation"] = "install"
    
    if st.button("Revert to AI Recommendations"):
        st.rerun()
    
    # Update costs for user selections
    selections = {
        "battery": selected_components["batteries"],
        "panel": selected_components["solarPanels"],
        "inverter": selected_components["inverters"],
        "charge_controller": selected_components["chargeControllers"],
        "battery_capacity_kwh": suggestion["battery_size"],
        "panel_capacity_kw": suggestion["production"] / suggestion["irradiance_forecast"]["yhat"].mean(),
        "inverter_capacity_kw": suggestion["inverter_capacity_kw"]
    }
    user_cost = calculate_cost(components_db, selections)
    st.metric("💵 Updated System Cost", f"{user_cost['avg']:,} XAF")

    # Component Details
    st.markdown("### 📋 Component Details")
    for comp_name in suggestion["components"]:
        found = False
        for category in components_db:
            if category != "installation":
                for component in components_db[category]:
                    if component["name"] == comp_name:
                        with st.expander(f"{comp_name} (Fitted)"):
                            st.write(f"**Tagline:** {component['tagline']}")
                            st.write(f"**Description:** {component['description']}")
                            st.write(f"**Best for:** {component['best_for']}")
                            st.write(f"**Lifespan:** {component['lifespan']}")
                            price_field = (
                                "price_per_kwh_xaf" if category == "batteries" else
                                "price_per_watt_xaf" if category == "solarPanels" else
                                "price_per_kw_xaf" if category == "inverters" else
                                "price_per_unit_xaf"
                            )
                            st.write(f"**Price Range:** {component[price_field]['min']:,.0f} - {component[price_field]['max']:,.0f} XAF per {'kWh' if category == 'batteries' else 'Watt' if category == 'solarPanels' else 'kW' if category == 'inverters' else 'unit'}")
                            st.write("**Pros:** " + ", ".join(component["pros"]))
                            st.write("**Cons:** " + ", ".join(component["cons"]))
                        found = True
                        break
            else:
                component = components_db[category]
                if component["name"] == comp_name:
                    with st.expander(f"{comp_name} (Fitted)"):
                        st.write(f"**Tagline:** {component['tagline']}")
                        st.write(f"**Description:** {component['description']}")
                        st.write(f"**Best for:** {component['best_for']}")
                        st.write(f"**Lifespan:** {component['lifespan']}")
                        st.write(f"**Price Range:** {component['price_per_system_xaf']['min']:,.0f} - {component['price_per_system_xaf']['max']:,.0f} XAF")
                        st.write("**Pros:** " + ", ".join(component["pros"]))
                        st.write("**Cons:** " + ", ".join(component["cons"]))
                    found = True
            if found:
                break
    
    # AI Energy Optimization Tips
    st.markdown("### 💡 AI Energy Optimization Tips")
    tips = generate_energy_tips(user_loads)
    for tip in tips:
        st.write(f"- {tip}")
    
    # Sensitivity Analysis
    st.markdown("### 📊 Sensitivity Analysis")
    budgets = [upper_budget - 200_000, upper_budget, upper_budget + 200_000]
    costs = []
    for b in budgets:
        if b >= 100_000:
            temp_suggestion = get_system_suggestion(region, daily_energy, b, solar_data, components_db, user_loads, install_type)
            costs.append(temp_suggestion["budget"])
        else:
            costs.append(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budgets, y=costs, mode="lines+markers", name="System Cost"))
    fig.update_layout(title="Cost vs. Budget", xaxis_title="Budget (XAF)", yaxis_title="Cost (XAF)", template="simple_white")
    st.plotly_chart(fig)

    # PDF Export
    if REPORTLAB_AVAILABLE:
        st.markdown("### 📄 Download Report")
        def generate_pdf():
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, "Solar Sizing Tool - Cameroon")
            c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            c.drawString(100, 710, f"Region: {region}")
            c.drawString(100, 690, f"Daily Energy Demand: {daily_energy:.2f} kWh/day")
            c.drawString(100, 670, f"Avg Predicted Irradiance: {suggestion['irradiance_forecast']['yhat'].mean():.2f} kWh/m²/day")
            c.drawString(100, 650, "System Components:")
            y = 630
            for cat, comp_id in selected_components.items():
                comp_name = (
                    components_db[cat]["name"] if cat == "installation"
                    else next(c["name"] for c in components_db[cat] if c["id"] == comp_id)
                )
                c.drawString(120, y, f"- {comp_name}")
                y -= 20
            c.drawString(100, y - 20, f"Total Cost: {user_cost['avg']:,} XAF")
            c.drawString(100, y - 40, "AI Energy Tips:")
            for i, tip in enumerate(tips):
                c.drawString(120, y - 60 - i * 20, f"- {tip}")
            c.showPage()
            c.save()
            buffer.seek(0)
            return buffer
        
        pdf_buffer = generate_pdf()
        st.download_button("Download PDF Report", pdf_buffer, file_name="solar_sizing_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export is disabled. Install 'reportlab' to enable this feature.")

    # Visual Charts
    st.markdown("### 📈 System Performance")
    def display_charts(production, demand, battery_kwh):
        hours = list(range(24))
        solar_profile = [0] * 24
        for h in range(6, 18):
            peak, val = 12, 0
            val = max(0, 1 - abs(h - peak) / 6) * production / 6
            solar_profile[h] = round(val, 2)
        battery_level = []
        charge = 0
        avg_demand = demand / 24
        for h in hours:
            charge += solar_profile[h] - avg_demand
            charge = max(0, min(charge, battery_kwh))
            battery_level.append(round(charge, 2))
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=hours, y=solar_profile, name="Solar Prod", marker_color="orange"))
        fig1.add_trace(go.Scatter(x=hours, y=[avg_demand]*24, name="Avg Demand", line=dict(color='red', dash='dash')))
        fig1.update_layout(title="🌞 Solar vs Demand", xaxis_title="Hour", yaxis_title="kWh", template="simple_white")
        st.plotly_chart(fig1)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hours, y=battery_level, name="Battery Level", fill='tozeroy', line_color='green'))
        fig2.update_layout(title="🔋 Battery Over 24h", xaxis_title="Hour", yaxis_title="kWh", template="simple_white")
        st.plotly_chart(fig2)

    display_charts(suggestion["production"], daily_energy, suggestion["battery_size"])

else:
    st.warning("Select at least one appliance to estimate your solar system.")

# Footer
st.markdown("---")
st.markdown("Built by **ProRivs** | 🚀 Powered by AI | 🇨🇲 Cameroon")