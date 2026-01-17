import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GSA Optimizer Pro", layout="wide")

# --- CORE LOGIC ---
class Stage:
    def __init__(self, name, process_time, holding_cost, capacity=None):
        self.name = name
        self.base_process_time = process_time
        self.effective_process_time = process_time
        self.holding_cost = holding_cost
        self.capacity = capacity if (capacity and capacity > 0) else None
        self.s_out = 0
        self.s_in = 0
        self.net_time = 0
        self.safety_stock = 0
        self.cost = 0

def calculate_queue_time(stage, mean_demand):
    if stage.capacity is None:
        return 0
    utilization = mean_demand / stage.capacity
    # Safety Cap at 99% to prevent infinite queue in simulation
    utilization = min(utilization, 0.99) 
    queue_time = stage.base_process_time * (utilization / (1 - utilization))
    return queue_time

def calculate_network(stages, demand_std, z_score, demand_mean):
    total_cost = 0
    current_inbound_time = 0 
    
    for i, stage in enumerate(stages):
        # 1. Apply Queue Time if constrained
        queue_delay = calculate_queue_time(stage, demand_mean)
        stage.effective_process_time = stage.base_process_time + queue_delay
        
        # 2. GSA Logic
        stage.s_in = current_inbound_time
        
        # Tau Calculation
        net_time = stage.s_in + stage.effective_process_time - stage.s_out
        stage.net_time = max(0, net_time)
        
        # Safety Stock Calculation
        stage.safety_stock = z_score * demand_std * np.sqrt(stage.net_time)
        stage.cost = stage.safety_stock * stage.holding_cost
        
        total_cost += stage.cost
        current_inbound_time = stage.s_out
        
    return total_cost

# --- THE UI ---
st.title("🏭 Multi-Echelon Safety Stock Simulator")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("1. Demand Settings")
    mean_demand = st.number_input("Avg Daily Demand", value=920)
    std_demand = st.number_input("Demand Std Dev", value=150)
    service_level = st.slider("Target Service Level (%)", 85, 99, 95)
    z_score = 1.645 

    st.markdown("---")
    st.header("2. Supply Chain Data")
    
    # Toggle between Manual and Import
    data_source = st.radio("Input Method:", ["Manual Entry", "Upload CSV"])

    # Initialize default values
    stages_data = []

    if data_source == "Manual Entry":
        st.subheader("Stage 1: Factory")
        f_t = st.number_input("Process Time", value=10, key="f_t")
        f_c = st.number_input("Holding Cost", value=5.0, key="f_c")
        f_cap = st.number_input("Capacity", value=1000, key="f_cap")
        stages_data.append({"Name": "Factory", "Time": f_t, "Cost": f_c, "Capacity": f_cap})

        st.subheader("Stage 2: Central DC")
        d_t = st.number_input("Process Time", value=5, key="d_t")
        d_c = st.number_input("Holding Cost", value=8.0, key="d_c")
        stages_data.append({"Name": "Central DC", "Time": d_t, "Cost": d_c, "Capacity": 0})

        st.subheader("Stage 3: Retailer")
        r_t = st.number_input("Process Time", value=1, key="r_t")
        r_c = st.number_input("Holding Cost", value=15.0, key="r_c")
        stages_data.append({"Name": "Retailer", "Time": r_t, "Cost": r_c, "Capacity": 0})

    else: # CSV Upload
        # Create a template for the user
        template_df = pd.DataFrame({
            "Stage Name": ["Factory", "Central DC", "Retailer"],
            "Process Time": [10, 5, 1],
            "Holding Cost": [5.0, 8.0, 15.0],
            "Capacity": [1000, 0, 0]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="gsa_template.csv",
            mime="text/csv",
        )
        
        uploaded_file = st.file_uploader("Upload your filled CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.dataframe(input_df.head(3))
                
                # Parse CSV into stages_data list
                for index, row in input_df.iterrows():
                    stages_data.append({
                        "Name": str(row[0]),
                        "Time": float(row[1]),
                        "Cost": float(row[2]),
                        "Capacity": float(row[3]) if pd.notna(row[3]) else 0
                    })
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- MAIN AREA ---

if len(stages_data) < 2:
    st.warning("Please upload data for at least 2 stages (e.g., Factory -> Retailer).")
    st.stop()

if st.button("🚀 Run Optimization"):
    
    # SETUP RANGES
    # For this demo, we optimize the FIRST stage (usually Factory) heavily, 
    # and the SECOND stage (DC) moderately. The LAST stage is always 0 service time (immediate to customer).
    
    # Dynamic ranges based on input lead times to make the graph look good
    max_lead_time = int(stages_data[0]["Time"] * 4) 
    factory_range = range(0, max_lead_time) 
    dc_range = range(0, 20)

    results = []
    best_cost = float('inf')
    best_config = None
    
    progress_bar = st.progress(0)
    
    # --- SIMULATION LOOP ---
    for i, f_time in enumerate(factory_range):
        local_min = float('inf')
        
        for dc_time in dc_range:
            # Reconstruct Stage Objects from data
            current_stages = []
            for idx, d in enumerate(stages_data):
                s = Stage(d["Name"], d["Time"], d["Cost"], d["Capacity"])
                
                # Apply simulated Service Times
                if idx == 0: s.s_out = f_time # Factory
                elif idx == 1: s.s_out = dc_time # DC
                else: s.s_out = 0 # Retailer/Last Node
                
                current_stages.append(s)
            
            # Calculate Total Cost
            cost = calculate_network(current_stages, std_demand, z_score, mean_demand)
            
            if cost < local_min:
                local_min = cost
            
            if cost < best_cost:
                best_cost = cost
                # Deep copy of state for result display
                best_config = []
                for s in current_stages:
                    # We create a simple dict to store the snapshot
                    best_config.append({
                        "Stage": s.name,
                        "Base Lead Time": s.base_process_time,
                        "Effective Time (w/ Queue)": round(s.effective_process_time, 1),
                        "Service Out (Promised)": s.s_out,
                        "Net Risk Time": round(s.net_time, 1),
                        "Safety Stock": round(s.safety_stock, 1),
                        "Cost": s.cost
                    })

        results.append(local_min)
        progress_bar.progress((i + 1) / len(factory_range))
        
    # --- DISPLAY RESULTS ---
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Optimization Curve")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(factory_range, results, color='#1f77b4', linewidth=2, label="Total Supply Chain Cost")
        
        # Highlight optimal point
        optimal_idx = results.index(best_cost)
        optimal_x = factory_range[optimal_idx]
        ax.plot(optimal_x, best_cost, 'ro', markersize=10, label="Optimal Point")
        
        ax.set_xlabel(f"{stages_data[0]['Name']} Outbound Service Time (Days)")
        ax.set_ylabel("Total Chain Cost ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.success(f"**Minimum Cost Found:**\n# ${best_cost:,.2f}")
        st.info(f"**Key Driver:**\nThe {stages_data[0]['Name']} should hold stock for **{optimal_x} days** before shipping.")

    st.subheader("Detailed Optimal Configuration")
    final_df = pd.DataFrame(best_config)
    
    # Format currency for display
    final_df["Cost"] = final_df["Cost"].map('${:,.2f}'.format)
    st.table(final_df)