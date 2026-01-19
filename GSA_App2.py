import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================================
# --- 1. CONFIGURATION & CLASSES ---
# =========================================

st.set_page_config(page_title="GSA Scenario Manager", layout="wide")

class Stage:
    def __init__(self, name, process_time, holding_cost, capacity=None):
        self.name = name
        self.base_process_time = process_time
        self.effective_process_time = process_time
        self.holding_cost = holding_cost
        # Treat 0 capacity as Infinite (None)
        self.capacity = capacity if (capacity and capacity > 0) else None
        self.s_out = 0
        self.s_in = 0
        self.net_time = 0
        self.safety_stock = 0
        self.cost = 0

def calculate_queue_time(stage, mean_demand):
    """Calculates Kingman's approximation for queue time if capacity is set."""
    if not stage.capacity: 
        return 0
    utilization = mean_demand / stage.capacity
    # Cap utilization at 99% to prevent math errors (divide by zero)
    if utilization >= 1.0: 
        utilization = 0.99
    
    # Formula: ProcessTime * (Utilization / (1 - Utilization))
    queue_time = stage.base_process_time * (utilization / (1 - utilization))
    return queue_time

def calculate_network(stages, demand_std, z_score, demand_mean):
    """Runs the GSA logic for the entire chain."""
    total_cost = 0
    total_stock = 0
    current_inbound_time = 0 
    
    for i, stage in enumerate(stages):
        # 1. Add queue time if factory is busy
        queue_delay = calculate_queue_time(stage, demand_mean)
        stage.effective_process_time = stage.base_process_time + queue_delay
        
        # 2. GSA Calculation
        stage.s_in = current_inbound_time
        
        # Net Risk Time = Inbound Service + Process Time - Outbound Service Promised
        net_time = stage.s_in + stage.effective_process_time - stage.s_out
        stage.net_time = max(0, net_time)
        
        # Safety Stock = k * sigma * sqrt(Tau)
        stage.safety_stock = z_score * demand_std * np.sqrt(stage.net_time)
        stage.cost = stage.safety_stock * stage.holding_cost
        
        total_cost += stage.cost
        total_stock += stage.safety_stock
        
        # Pass outbound time to next node
        current_inbound_time = stage.s_out
        
    return total_cost, total_stock

# =========================================
# --- 2. UI LAYOUT ---
# =========================================

st.title("🏭 Multi-Scenario Safety Stock Simulator")

# --- VISUAL DIAGRAM ---
st.subheader("Supply Chain Visualizer")
st.graphviz_chart("""
    digraph G {
        rankdir=LR; 
        node [shape=box, style="filled,rounded", fontname="Arial", margin="0.2,0.1"];
        edge [fontname="Arial", fontsize=10, color="#666666"];
        Factory [fillcolor="#e3f2fd", label="🏭 Factory", color="#1565c0", penwidth=2];
        DC [fillcolor="#fff9c4", label="🚚 Central DC", color="#fbc02d", penwidth=2];
        Retailer [fillcolor="#ffe0b2", label="🏪 Retailer", color="#e65100", penwidth=2];
        
        Factory -> DC [label=" T=Lead Time"];
        DC -> Retailer [label=" T=Lead Time"];
    }
""")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Demand Profile")
    # UPDATED: Default value is now 1000
    mean_demand = st.number_input("Avg Daily Demand", value=1000, help="Units/Day")
    std_demand = st.number_input("Demand Std Dev", value=150, help="Daily Variability")
    
    st.markdown("---")
    st.header("2. Scenario Selector")
    # Users can pick multiple service levels to compare
    selected_levels = st.multiselect(
        "Select Service Levels to Compare (Max 5)",
        options=[85, 90, 92, 94, 95, 96, 97, 98, 99, 99.5, 99.9],
        default=[90, 95, 99]
    )
    
    if len(selected_levels) > 5:
        st.error("Please select a maximum of 5 scenarios to keep the graph readable.")
        st.stop()
    
    if not selected_levels:
        st.warning("Please select at least one service level.")
        st.stop()
        
    st.markdown("---")
    data_source = st.radio("Input Method:", ["Manual Entry", "Upload CSV"])

# --- DATA INPUTS ---
stages_data = []

if data_source == "Manual Entry":
    st.caption("Enter parameters matching the diagram above. Set Capacity to 0 for infinite capacity.")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🏭 Factory")
        f_t = st.number_input("Lead-time (Days)", value=10, key="f_t")
        f_c = st.number_input("Holding Cost ($)", value=5.0, key="f_c")
        # FIXED: Added min_value=0 to prevent negative inputs
        f_cap = st.number_input("Capacity (0=Inf)", value=0, min_value=0, key="f_cap", help="Max units/day")
        stages_data.append({"Name": "Factory", "Time": f_t, "Cost": f_c, "Capacity": f_cap})
        
    with c2:
        st.markdown("### 🚚 Central DC")
        d_t = st.number_input("Lead-time (Days)", value=5, key="d_t")
        d_c = st.number_input("Holding Cost ($)", value=8.0, key="d_c")
        stages_data.append({"Name": "Central DC", "Time": d_t, "Cost": d_c, "Capacity": 0})
        
    with c3:
        st.markdown("### 🏪 Retailer")
        r_t = st.number_input("Lead-time (Days)", value=1, key="r_t")
        r_c = st.number_input("Holding Cost ($)", value=15.0, key="r_c")
        stages_data.append({"Name": "Retailer", "Time": r_t, "Cost": r_c, "Capacity": 0})

else:
    # CSV UPLOAD LOGIC
    st.info("Upload your CSV file using the template in the sidebar.")
    with st.sidebar:
        template_df = pd.DataFrame({
            "Stage Name": ["Factory", "Central DC", "Retailer"],
            "Lead-time (Days)": [10, 5, 1],
            "Holding Cost": [5.0, 8.0, 15.0],
            "Capacity": [0, 0, 0]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Template", data=csv_template, file_name="gsa_template.csv", mime="text/csv")
        
        uploaded_file = st.file_uploader("Upload filled CSV", type=["csv"])
        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                for index, row in input_df.iterrows():
                    stages_data.append({
                        "Name": str(row[0]),
                        "Time": float(row[1]),
                        "Cost": float(row[2]),
                        "Capacity": float(row[3]) if pd.notna(row[3]) else 0
                    })
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

# =========================================
# --- 3. OPTIMIZATION ENGINE ---
# =========================================

# UPDATED: Button moved here, immediately below the input fields
st.markdown("---")
if len(stages_data) >= 2 and st.button("🚀 Run Scenario Comparison", type="primary"):
    
    # 1. Define Search Ranges (Optimization Space)
    # We search Factory times 0 to 4x its lead time
    max_lead_time = int(stages_data[0]["Time"] * 4) 
    if max_lead_time < 5: max_lead_time = 10
    
    factory_range = range(0, max_lead_time) 
    dc_range = range(0, 20)

    # Dictionary to store results for all scenarios
    all_scenarios = {} 
    
    with st.spinner("Simulating supply chain scenarios..."):
        progress_bar = st.progress(0)
        
        # --- LOOP 1: Iterate through User Selected Service Levels ---
        for idx, svc_lvl in enumerate(selected_levels):
            
            # Calculate Z for this specific scenario
            z = norm.ppf(svc_lvl / 100.0)
            
            scenario_cost_curve = []
            best_cost = float('inf')
            best_total_stock = 0
            best_config_data = [] # Stores table data for tabs
            
            # --- LOOP 2: Iterate Factory Service Times ---
            for f_time in factory_range:
                local_min = float('inf')
                
                # --- LOOP 3: Iterate DC Service Times ---
                for dc_time in dc_range:
                    # Re-create stage objects fresh for every calc
                    current_stages = []
                    for stage_idx, d in enumerate(stages_data):
                        s = Stage(d["Name"], d["Time"], d["Cost"], d["Capacity"])
                        if stage_idx == 0: s.s_out = f_time
                        elif stage_idx == 1: s.s_out = dc_time
                        else: s.s_out = 0 # Retailer
                        current_stages.append(s)
                    
                    # Run Math
                    cost, total_stock = calculate_network(current_stages, std_demand, z, mean_demand)
                    
                    if cost < local_min: 
                        local_min = cost
                    
                    # Track Global Best for this Scenario
                    if cost < best_cost:
                        best_cost = cost
                        best_total_stock = total_stock
                        
                        # Save detailed data for the table later
                        best_config_data = []
                        for s in current_stages:
                            best_config_data.append({
                                "Stage": s.name,
                                "Lead-time": s.base_process_time,
                                "Promised Out": s.s_out,
                                "Safety Stock": s.safety_stock,
                                "Cost": s.cost
                            })
                            
                scenario_cost_curve.append(local_min)
            
            # Save all data for this Service Level
            all_scenarios[svc_lvl] = {
                "costs": scenario_cost_curve,
                "best_cost": best_cost,
                "best_stock": best_total_stock,
                "z": z,
                "config": best_config_data
            }
            
            # Update progress bar
            progress_bar.progress((idx + 1) / len(selected_levels))

    # =========================================
    # --- 4. RESULTS DASHBOARD ---
    # =========================================
    
    # LAYOUT: Graph on Left (2/3), Summary on Right (1/3)
    col_graph, col_summary = st.columns([2, 1])
    
    # --- A. COMPARATIVE GRAPH ---
    with col_graph:
        st.subheader("Cost Comparison Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors for lines
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, svc in enumerate(selected_levels):
            data = all_scenarios[svc]
            ax.plot(factory_range, data["costs"], label=f"{svc}% Service", color=colors[i%6], linewidth=2)
            
            # Mark the optimal point (lowest cost)
            opt_idx = data["costs"].index(data["best_cost"])
            ax.plot(factory_range[opt_idx], data["best_cost"], 'o', color=colors[i%6], markersize=8, markeredgecolor='white')

        ax.set_xlabel(f"{stages_data[0]['Name']} Outbound Service Time (Days)")
        ax.set_ylabel("Total Inventory Cost ($)")
        ax.set_title("Efficient Frontier: Cost vs. Factory Speed")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # --- B. GRID SUMMARY & RECOMMENDATION ---
    with col_summary:
        st.subheader("Scenario Summary")
        
        summary_rows = []
        for svc in selected_levels:
            d = all_scenarios[svc]
            summary_rows.append({
                "Service Level": f"{svc}%",
                "Total Cost": d["best_cost"],
                "Total Units": d["best_stock"],
                "Z-Score": f"{d['z']:.2f}"
            })
            
        df_sum = pd.DataFrame(summary_rows)
        
        # FIXED: replaced use_container_width=True with width='stretch'
        st.dataframe(
            df_sum,
            column_config={
                "Total Cost": st.column_config.NumberColumn(format="$%.2f"),
                "Total Units": st.column_config.NumberColumn(format="%.0f")
            },
            hide_index=True,
            width='stretch' 
        )
        
        # Recommendation Engine
        st.subheader("💡 Recommendation")
        
        # Find min and max cost scenarios
        sorted_scenarios = sorted(summary_rows, key=lambda x: x["Total Cost"])
        cheapest = sorted_scenarios[0]
        most_expensive = sorted_scenarios[-1]
        
        diff_cost = most_expensive["Total Cost"] - cheapest["Total Cost"]
        if cheapest["Total Cost"] > 0:
            pct_diff = (diff_cost / cheapest["Total Cost"]) * 100
        else:
            pct_diff = 0
        
        if len(selected_levels) > 1:
            st.info(f"""
            **Cost Sensitivity:** Moving from **{cheapest['Service Level']}** to **{most_expensive['Service Level']}** increases your inventory investment by **${diff_cost:,.0f}** (+{pct_diff:.0f}%).
            
            **Strategic Question:**
            Is the extra service worth the additional ${diff_cost:,.0f} in holding costs?
            """)
        else:
            st.info("Run multiple scenarios to see a cost comparison recommendation.")

    # --- C. DETAILED BREAKDOWN TABS ---
    st.subheader("Detailed Configuration by Scenario")
    tabs = st.tabs([f"{s}% Level" for s in selected_levels])
    
    for i, svc in enumerate(selected_levels):
        with tabs[i]:
            cfg = all_scenarios[svc]["config"]
            df_detail = pd.DataFrame(cfg)
            
            # FIXED: replaced use_container_width=True with width='stretch'
            st.dataframe(
                df_detail,
                column_config={
                    "Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Safety Stock": st.column_config.NumberColumn(format="%.1f"),
                    "Promised Out": st.column_config.NumberColumn(help="Optimized Outbound Service Time"),
                },
                hide_index=True,
                width='stretch'
            )