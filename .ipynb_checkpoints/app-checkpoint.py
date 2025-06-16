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
    st.warning("PDF export feature is unavailable because 'reportlab' package is not installed. Install it using 'pip install reportlab'.")

st.set_page_config(page_title="Solar Sizing Tool", layout="centered")

# Load Data Files
solar_data = load_solar_irradiance()
components_db = load_components()
load_prices = load_appliances()

# Title & Intro
st.title("☀️ Solar Sizing Tool (Cameroon)")
st.markdown("Optimize your off-grid or hybrid solar system with AI-driven recommendations.")
st.markdown("---")

# Inputs
st.markdown("## 🗺️ Location & Budget Input")
col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("📍 Select Your Region", list(solar_data.keys()))
    install_type = st.selectbox("🏠 Installation Type", ["urban", "rural"])
    minimalist_mode = st.checkbox("🌟 Minimalist System (e.g., for lighting only)", value=False)
    if minimalist_mode:
        use_kit = st.radio("Select System Type", ["Solar Kit", "Custom Components"], index=0)
        is_dc_only = st.checkbox("DC-Only System (cheapest for lighting)", value=False)
    else:
        use_kit = False
        is_dc_only = False

with col2:
    upper_budget = st.slider("💰 Max Budget (XAF)", min_value=50_000, max_value=1_500_000, step=10_000, value=100_000)
    inverter_type = st.selectbox("🔌 Hybrid Inverter Type", ["PWM", "MPPT"])

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
    daily_energy, apparent_power_kva = calculate_energy_demand(user_loads)
    st.success(f"🔋 Daily Demand: **{daily_energy:.2f} kWh/day**")
    st.write(f"**Apparent Power**: {apparent_power_kva:.2f} kVA (for inverter sizing)")
    
    if daily_energy < 0.5 and not minimalist_mode:
        st.warning("Your energy demand is very low (e.g., suitable for lighting). Enable 'Minimalist System' for a more cost-effective solution.")
    
    suggestion = get_system_suggestion(
        region, daily_energy, upper_budget, solar_data, components_db, user_loads,
        install_type, inverter_type, minimalist_mode, use_kit, is_dc_only
    )
    
    if suggestion['budget'] > daily_energy * 500_000 and daily_energy < 0.5:
        st.warning(f"The system cost ({suggestion['budget']:,} XAF) seems high for your low demand ({daily_energy:.2f} kWh/day). Consider enabling 'Minimalist System' or using energy-efficient bulbs.")
    
    st.markdown("## 🔍 System Recommendation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔋 Battery Size", f"{suggestion['battery_size']} kWh")
    c2.metric("☀️ Avg Daily Output", f"{suggestion['production']} kWh")
    c3.metric("💵 Total System Cost", f"{suggestion['budget']:,} XAF")
    c4.metric("🌟 Reliability Score", f"{suggestion['reliability_score']:.1f}/10")
    
    st.markdown("### 📈 Energy Production vs. Demand (7 Days)")
    combined_fig = go.Figure()
    combined_fig.add_trace(go.Scatter(
        x=suggestion["production_forecast"]["ds"],
        y=suggestion["production_forecast"]["yhat"],
        mode="lines+markers",
        name="Energy Production",
        line=dict(color="orange")
    ))
    combined_fig.add_trace(go.Scatter(
        x=suggestion["production_forecast"]["ds"],
        y=[daily_energy]*7,
        mode="lines",
        name="Energy Demand",
        line=dict(color="red", dash="dash")
    ))
    combined_fig.update_layout(
        title="Energy Production vs. Demand",
        xaxis_title="Date",
        yaxis_title="kWh/day",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(combined_fig)

    # Cost Comparison
    if minimalist_mode:
        st.markdown("### 💰 Cost Comparison")
        kit_suggestion = get_system_suggestion(
            region, daily_energy, upper_budget, solar_data, components_db, user_loads,
            install_type, inverter_type, minimalist_mode, True, is_dc_only
        )
        custom_suggestion = get_system_suggestion(
            region, daily_energy, upper_budget, solar_data, components_db, user_loads,
            install_type, inverter_type, minimalist_mode, False, is_dc_only
        )
        comparison_table = pd.DataFrame({
            "Option": ["Solar Kit", "Custom Components"],
            "Cost (XAF)": [kit_suggestion["budget"], custom_suggestion["budget"]],
            "Components": [
                ", ".join(kit_suggestion["components"]),
                ", ".join(custom_suggestion["components"])
            ]
        })
        st.table(comparison_table)

    st.markdown("### 🧰 Component Selection")
    selected_components = {}
    if "solarKits" in suggestion["explanations"]:
        options = [comp["name"] for comp in components_db["solarKits"] if comp["is_dc_only"] == is_dc_only]
        default = suggestion["components"][0]
        # Validate default kit
        if not options:
            st.warning("No solar kits available for the selected mode. Falling back to custom components.")
            selected_components["solarKits"] = None
        else:
            if default not in options:
                default = options[0]  # Fall back to first available kit
                st.info(f"Recommended kit '{suggestion['components'][0]}' not found. Defaulting to '{default}'.")
            selected = st.selectbox("Select Solar Kit", options, index=options.index(default), key="select_solarKits")
            selected_components["solarKits"] = next(c["id"] for c in components_db["solarKits"] if c["name"] == selected)
            st.write(f"**AI Explanation**: {suggestion['explanations']['solarKits']}")
            selected_components["installation"] = "install"
    else:
        for category in ["batteries", "solarPanels", "inverters"]:
            if category == "inverters" and is_dc_only:
                selected_components[category] = None
                continue
            options = [comp["name"] for comp in components_db[category]]
            default = next((c["name"] for c in components_db[category] if c["name"] in suggestion["components"]), options[0])
            selected = st.selectbox(f"Select {category.title()}", options, index=options.index(default), key=f"select_{category}")
            selected_components[category] = next(c["id"] for c in components_db[category] if c["name"] == selected)
            st.write(f"**AI Explanation**: {suggestion['explanations'][category]}")
        selected_components["installation"] = "install"
    
    if st.button("Revert to AI Recommendations"):
        st.rerun()
    
    selections = {
        "battery": selected_components.get("batteries"),
        "panel": selected_components.get("solarPanels"),
        "inverter": selected_components.get("inverters"),
        "solarKits": selected_components.get("solarKits"),
        "battery_capacity_kwh": suggestion["battery_size"],
        "panel_capacity_kw": suggestion["production"] / suggestion["irradiance_forecast"]["yhat"].mean(),
        "inverter_capacity_kw": suggestion["inverter_capacity_kw"]
    }
    user_cost = calculate_cost(components_db, selections, is_dc_only)
    
    st.markdown("### 💵 Cost Breakdown")
    if "solarKits" in selected_components and selected_components["solarKits"]:
        cost_table = pd.DataFrame({
            "Component": ["Solar Kit", "Installation"],
            "Cost (XAF)": [
                user_cost["components"]["kit"],
                user_cost["components"]["installation"]
            ]
        })
    else:
        components = ["Battery", "Solar Panels"]
        costs = [user_cost["components"]["battery"], user_cost["components"]["panel"]]
        if not is_dc_only:
            components.append("Inverter")
            costs.append(user_cost["components"]["inverter"])
        components.append("Installation")
        costs.append(user_cost["components"]["installation"])
        cost_table = pd.DataFrame({
            "Component": components,
            "Cost (XAF)": costs
        })
    st.table(cost_table)
    st.metric("💵 Total System Cost", f"{user_cost['avg']:,} XAF")
    if minimalist_mode:
        st.info("Minimalist System enabled: Optimized for low-demand applications like lighting.")
    if is_dc_only:
        st.info("DC-Only System enabled: Cheapest option for lighting, no inverter required.")

    st.markdown("### 📋 Component Details")
    for comp_name in suggestion["components"]:
        found = False
        for category in components_db:
            if category != "installation":
                for component in components_db[category]:
                    if component["name"] == comp_name:
                        with st.expander(f"{comp_name} (Fitted)"):
                            st.write(f"**Tagline**: {component['tagline']}")
                            st.write(f"**Description**: {component['description']}")
                            st.write(f"**Best for**: {component['best_for']}")
                            st.write(f"**Lifespan**: {component['lifespan']}")
                            price_field = (
                                "price_per_kwh_xaf" if category == "batteries" else
                                "price_per_watt_xaf" if category == "solarPanels" else
                                "price_per_kw_xaf" if category == "inverters" else
                                "price_per_system_xaf"
                            )
                            st.write(f"**Price Range**: {component[price_field]['min']:,.0f} - {component[price_field]['max']:,.0f} XAF per {'kWh' if category == 'batteries' else 'Watt' if category == 'solarPanels' else 'kW' if category == 'inverters' else 'system'}")
                            st.write("**Pros**: " + ", ".join(component["pros"]))
                            st.write("**Cons**: " + ", ".join(component["cons"]))
                        found = True
                        break
            else:
                component = components_db[category]
                if component["name"] == comp_name:
                    with st.expander(f"{comp_name} (Fitted)"):
                        st.write(f"**Tagline**: {component['tagline']}")
                        st.write(f"**Description**: {component['description']}")
                        st.write(f"**Best for**: {component['best_for']}")
                        st.write(f"**Lifespan**: {component['lifespan']}")
                        st.write(f"**Price Range**: {component['price_per_system_xaf']['min']:,.0f} - {component['price_per_system_xaf']['max']:,.0f} XAF")
                        st.write("**Pros**: " + ", ".join(component["pros"]))
                        st.write("**Cons**: " + ", ".join(component["cons"]))
                    found = True
            if found:
                break
    
    st.markdown("### 💡 AI Energy Optimization Tips")
    tips = generate_energy_tips(user_loads)
    for tip in tips:
        st.write(f"- {tip}")
    
    st.markdown("### 📊 Sensitivity Analysis")
    budgets = [upper_budget * (1 + i * 0.2) for i in range(-2, 3)]
    costs = []
    reliability_scores = []
    for b in budgets:
        if b >= 50_000:
            temp_suggestion = get_system_suggestion(
                region, daily_energy, b, solar_data, components_db, user_loads,
                install_type, inverter_type, minimalist_mode, use_kit, is_dc_only
            )
            costs.append(temp_suggestion["budget"])
            reliability_scores.append(temp_suggestion["reliability_score"])
        else:
            costs.append(None)
            reliability_scores.append(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budgets, y=costs, mode="lines+markers", name="System Cost", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=budgets, y=reliability_scores, mode="lines+markers", name="Reliability Score", yaxis="y2", line=dict(color="green")))
    fig.update_layout(
        title="Cost and Reliability vs. Budget",
        xaxis_title="Budget (XAF)",
        yaxis=dict(title="Cost (XAF)", side="left"),
        yaxis2=dict(title="Reliability Score", side="right", overlaying="y"),
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig)

    if REPORTLAB_AVAILABLE:
        st.markdown("### 📄 Download Report")
        def generate_pdf():
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(100, 750, "Solar Sizing Tool - Cameroon")
            c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
            c.drawString(100, 710, f"Region: {region}")
            c.drawString(100, 690, f"Daily Energy Demand: {daily_energy:.2f} kWh/day")
            c.drawString(100, 670, f"Average Predicted Forecast: {suggestion['irradiance_forecast']['yhat'].mean():,.2f} kWh/m²/day")
            c.drawString(100, 650, f"Reliability Score: {suggestion['reliability_score']:.1f}/10")
            c.drawString(100, 630, f"System Mode: {'Minimalist' if minimalist_mode else 'Standard'} ({'DC-Only' if is_dc_only else 'AC/DC'})")
            c.drawString(100, 610, "System Components:")
            y = 590
            cost_key_map = {
                "batteries": "battery",
                "solarPanels": "panel",
                "inverters": "inverter",
                "installation": "installation",
                "solarKits": "kit"
            }
            for cat, comp_id in selected_components.items():
                if comp_id:
                    if cat == "installation":
                        comp_name = components_db[cat]["name"]
                    else:
                        comp_name = next(c["name"] for c in components_db[cat] if c["id"] == comp_id)
                    cost_key = cost_key_map.get(cat, "kit")
                    c.drawString(120, y, f"- {comp_name}: {user_cost['components'][cost_key]:,.0f} XAF")
                    y -= 20
            c.drawString(100, y - 20, f"Total Cost: {user_cost['avg']:,} XAF")
            c.drawString(100, y - 40, "AI Energy Optimization Tips:")
            for i, tip in enumerate(tips[:5]):
                c.drawString(120, y - 60 - i * 20, f"- {tip[:80]}")
            c.showPage()
            c.save()
            buffer.seek(0)
            return buffer
        
        pdf_buffer = generate_pdf()
        st.download_button("Download PDF Report", pdf_buffer, file_name="solar_sizing_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export is disabled. Install 'reportlab' to enable this feature.")

    st.markdown("### 📈 System Performance")
    def display_charts(production, demand, battery_kwh):
        hours = list(range(24))
        solar_profile = [0] * 24
        for h in range(6, 18):
            peak, val = 12, 0
            val = max(0, 1 - abs(h - peak) / 6) * production / 6
            solar_profile[h] = round(val, 2)
        battery_level = []
        charge = battery_kwh * 0.5
        avg_demand = demand / 24
        for h in hours:
            charge += solar_profile[h] - avg_demand
            charge = max(0, min(charge, battery_kwh))
            battery_level.append(round(charge, 2))
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=hours, y=solar_profile, name="Solar Production", line=dict(color="orange")))
        fig1.add_trace(go.Scatter(x=hours, y=[avg_demand]*24, name="Avg Demand", line=dict(color="red", dash="dash")))
        fig1.update_layout(
            title="🌞 Solar Production vs. Demand (24h)",
            xaxis_title="Hour",
            yaxis_title="kWh",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig1)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hours, y=battery_level, name="Battery Level", fill="tozeroy", line=dict(color="green")))
        fig2.update_layout(
            title="🔋 Battery State of Charge (24h)",
            xaxis_title="Hour",
            yaxis_title="kWh",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig2)

    display_charts(suggestion["production"], daily_energy, suggestion["battery_size"])

else:
    st.warning("Select at least one appliance to estimate your solar system.")

st.markdown("---")
st.markdown("Built by **SunSmart** | 🚀 Powered by AI | 🇨🇲 Cameroon")