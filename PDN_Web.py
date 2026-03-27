import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# ==========================================
# 1. Page Configuration & Initialization
# ==========================================
st.set_page_config(page_title="PDN Impedance Simulator", layout="wide", page_icon="⚡")
st.title("⚡ Full System PDN Impedance Simulator")
st.markdown("Built with **Continuous PSO** and **Plotly**. Supports **Impedance Masking** and **Targeted Resonance Suppression**. *(Powered by 3D Tensor Vectorization Engine)*")

def auto_format_cap(val_f):
    if val_f < 1e-9: return round(float(val_f * 1e12), 2), "pF"
    elif val_f < 1e-6: return round(float(val_f * 1e9), 2), "nF"
    else: return round(float(val_f * 1e6), 2), "uF"

default_values = {
    "target_z": "10 (Standard IC)",
    "L_mm": 30.0, "W_mm": 10.0, "Er": 3.8, "d_mil": 2.0, "I_op": 10.0,
    "ESL": 0.1, "L_cap_via": 0.2, "L_ic_via": 0.1, "L_pkg": 0.5, "C_die": 200.0,
    "num_caps": 5,
    "C1_val": 1.0, "C1_unit": "nF", "C2_val": 10.0, "C2_unit": "nF",
    "C3_val": 1.0, "C3_unit": "uF", "C4_val": 10.0, "C4_unit": "uF",
    "C5_val": 47.0, "C5_unit": "uF", "run_tune": False,
    "use_band": True, "f_min": 1.0, "f_max": 20.0, "mask_target_z": 12.5,
    "use_target": False, "target_freq": 15.0, "target_weight": 50
}
for key, value in default_values.items():
    if key not in st.session_state: st.session_state[key] = value

# ==========================================
# 2. Core Engine (☢️ 3D Tensor Batched & Decimated)
# ==========================================
@st.cache_data(show_spinner=False)
def calc_core(L, W, Er, d, I_op, num_caps, C1, C2, C3, C4, C5, ESL_val, L_cap_via, L_ic_via, L_pkg, C_die, frequencies):
    Nx, Ny = 5, 5
    N_total = Nx * Ny
    dx, dy = L / Nx, W / Ny
    t_cu = 3.5e-5        
    E0 = 8.854e-12
    U0 = 4 * np.pi * 1e-7
    rho_cu = 1.68e-8     
    
    R_sheet = rho_cu / t_cu; R_x = R_sheet * (dx / dy); R_y = R_sheet * (dy / dx)  
    L_cell = U0 * d; C_cell = Er * E0 * (dx * dy) / d
    
    vrm_node = 2   
    ic_node = 22   
    
    cap_config = []
    if num_caps >= 1: cap_config.append({'node': 16, 'C': C1})
    if num_caps >= 2: cap_config.append({'node': 18, 'C': C2})
    if num_caps >= 3: cap_config.append({'node': 11, 'C': C3})
    if num_caps >= 4: cap_config.append({'node': 13, 'C': C4})
    if num_caps >= 5: cap_config.append({'node': 7,  'C': C5})
        
    M_C = np.eye(N_total)
    M_x = np.zeros((N_total, N_total)); M_y = np.zeros((N_total, N_total))

    for i in range(Nx):
        for j in range(Ny):
            n = i * Ny + j
            if i < Nx - 1:
                nr = (i + 1) * Ny + j
                M_x[n, n] += 1; M_x[nr, nr] += 1; M_x[n, nr] -= 1; M_x[nr, n] -= 1
            if j < Ny - 1:
                nu = i * Ny + (j + 1)
                M_y[n, n] += 1; M_y[nu, nu] += 1; M_y[n, nu] -= 1; M_y[nu, n] -= 1

    G_matrix = M_x * (1.0 / R_x) + M_y * (1.0 / R_y)
    G_matrix[vrm_node, vrm_node] += 1e9 
    I_dc = np.zeros(N_total); I_dc[ic_node] = -I_op 
    V_dc = np.linalg.solve(G_matrix, I_dc)
    ir_drop_v = 0.0 - V_dc[ic_node] 
    
    N_f = len(frequencies)
    w = 2 * np.pi * frequencies
    Y_C_arr = 1j * w * C_cell
    Y_x_arr = 1.0 / (R_x + 1j * w * L_cell)
    Y_y_arr = 1.0 / (R_y + 1j * w * L_cell)

    Y_matrix_batch = (
        Y_C_arr[:, None, None] * M_C[None, :, :] +
        Y_x_arr[:, None, None] * M_x[None, :, :] +
        Y_y_arr[:, None, None] * M_y[None, :, :]
    )
    Y_matrix_batch[:, vrm_node, vrm_node] += 1e9

    ESR = 5e-3; R_die = 1e-3
    for cap in cap_config:
        Total_L = ESL_val + L_cap_via
        Y_cap_arr = 1.0 / (ESR + 1j * w * Total_L + 1.0 / (1j * w * cap['C']))
        Y_matrix_batch[:, cap['node'], cap['node']] += Y_cap_arr

    # 🛠️【修復點】：改用單一 1D 向量，讓 NumPy 自動完美廣播 (Broadcasting)
    I_ac = np.zeros(N_total, dtype=complex)
    I_ac[ic_node] = 1.0 + 0j

    # 一行 LAPACK 解決所有頻率點！
    V_ac_batch = np.linalg.solve(Y_matrix_batch, I_ac)

    Z_pcb_pin = V_ac_batch[:, ic_node] + 1j * w * L_ic_via
    Z_profile_pcb = np.abs(Z_pcb_pin)

    Z_series = Z_pcb_pin + 1j * w * L_pkg
    Z_cdie = R_die + np.where(C_die > 0, 1.0 / (1j * w * C_die + 1e-15), 1e15)
    Z_die_total = np.where(C_die > 0, (Z_series * Z_cdie) / (Z_series + Z_cdie), Z_series)
    Z_profile_die = np.abs(Z_die_total)
        
    return Z_profile_pcb, Z_profile_die, ir_drop_v

def get_cap_val(val, unit):
    if unit == "uF": return val * 1e-6
    elif unit == "nF": return val * 1e-9
    elif unit == "pF": return val * 1e-12
    return val * 1e-6

# ==========================================
# 3. Continuous PSO Algorithm
# ==========================================
def continuous_pso(cost_func, bounds, num_particles=40, maxiter=40, progress_bar=None):
    dim = len(bounds)
    positions = np.zeros((num_particles, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], num_particles)
        
    velocities = np.zeros((num_particles, dim))
    pbest_pos = positions.copy()
    pbest_scores = np.array([cost_func(p) for p in positions])
    
    gbest_idx = np.argmin(pbest_scores)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    w, c1, c2 = 0.6, 1.5, 1.5 
    for step in range(maxiter):
        r1, r2 = np.random.rand(num_particles, dim), np.random.rand(num_particles, dim)
        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        new_positions = positions + velocities
        for i in range(dim): new_positions[:, i] = np.clip(new_positions[:, i], bounds[i][0], bounds[i][1])
            
        positions = new_positions
        scores = np.array([cost_func(p) for p in positions])
        
        better_mask = scores < pbest_scores
        pbest_scores[better_mask] = scores[better_mask]
        pbest_pos[better_mask] = positions[better_mask]
        
        min_score_idx = np.argmin(pbest_scores)
        if pbest_scores[min_score_idx] < gbest_score:
            gbest_score = pbest_scores[min_score_idx]
            gbest_pos = pbest_pos[min_score_idx].copy()
            
        if progress_bar: progress_bar.progress((step + 1) / maxiter)
    return gbest_pos

# ==========================================
# 4. PSO Intercept & Cost Function
# ==========================================
if st.session_state.run_tune:
    st.info("🦅 **Running Continuous PSO...** (AI is searching at extreme vectorized speed)")
    progress_bar = st.progress(0.0)
    
    if st.session_state.use_band:
        target_z_ohm = st.session_state.mask_target_z / 1000.0
    else:
        match = re.search(r"(\d+(\.\d+)?)", st.session_state.target_z)
        target_z_ohm = (float(match.group(1)) if match else 100.0) / 1000.0
        
    frequencies_opt = np.logspace(4, 9.5, 80) 
    L_fixed, W_fixed = st.session_state.L_mm * 1e-3, st.session_state.W_mm * 1e-3
    ESL_fixed = st.session_state.ESL * 1e-9
    
    bounds = [(1.0, 10000.0), (1.0, 1000.0), (0.1, 10.0), (1.0, 100.0), (10.0, 1000.0)]
    
    def cost_func(cap_array):
        C1, C2 = cap_array[0] * 1e-12, cap_array[1] * 1e-9
        C3, C4, C5 = cap_array[2] * 1e-6, cap_array[3] * 1e-6, cap_array[4] * 1e-6
        try:
            _, Z_die, ir_drop = calc_core(
                L_fixed, W_fixed, st.session_state.Er, st.session_state.d_mil * 25.4e-6, st.session_state.I_op, 
                st.session_state.num_caps, C1, C2, C3, C4, C5, ESL_fixed, 
                st.session_state.L_cap_via * 1e-9, st.session_state.L_ic_via * 1e-9, 
                st.session_state.L_pkg * 1e-9, st.session_state.C_die * 1e-9, frequencies_opt)
            
            excess = np.maximum(0, Z_die - target_z_ohm)
            weight_mask = np.ones_like(frequencies_opt)
            
            if st.session_state.use_band:
                f_min_hz = st.session_state.f_min * 1e6
                f_max_hz = st.session_state.f_max * 1e6
                out_of_band = (frequencies_opt < f_min_hz) | (frequencies_opt > f_max_hz)
                excess[out_of_band] = 0.0 
                
            if st.session_state.use_target:
                f_target = st.session_state.target_freq * 1e6
                sigma = f_target * 0.25 
                weight_mask += st.session_state.target_weight * np.exp(-0.5 * ((frequencies_opt - f_target) / sigma)**2)
                
            penalty = 0.0
            if ir_drop > 0.05: penalty += (ir_drop - 0.05) * 1e5
            
            weighted_excess = excess * weight_mask
            penalty += np.sum(weighted_excess**2) * 1e6 + np.max(weighted_excess) * 1e7 
            return penalty
        except:
            return 1e9

    best_caps = continuous_pso(cost_func, bounds, num_particles=40, maxiter=40, progress_bar=progress_bar)
    
    c1_v, c1_u = auto_format_cap(best_caps[0] * 1e-12)
    c2_v, c2_u = auto_format_cap(best_caps[1] * 1e-9)
    c3_v, c3_u = auto_format_cap(best_caps[2] * 1e-6)
    c4_v, c4_u = auto_format_cap(best_caps[3] * 1e-6)
    c5_v, c5_u = auto_format_cap(best_caps[4] * 1e-6)
    
    st.session_state.update({
        "C1_val": c1_v, "C1_unit": c1_u, "C2_val": c2_v, "C2_unit": c2_u,
        "C3_val": c3_v, "C3_unit": c3_u, "C4_val": c4_v, "C4_unit": c4_u,
        "C5_val": c5_v, "C5_unit": c5_u, "run_tune": False
    })

# ==========================================
# 5. Sidebar UI Design
# ==========================================
with st.sidebar:
    st.header("⚙️ Global Specifications")
    st.selectbox("🎯 Target Impedance (Target Z):", ["100 (General)", "10 (Standard IC)", "5 (DDR5)", "1 (CPU Core)"], key="target_z")
    
    st.divider()
    st.markdown("**【 📊 Datasheet Band Limit 】**")
    st.checkbox("Enable Impedance Mask", key="use_band")
    if st.session_state.use_band:
        st.number_input("Target Z in this band (mΩ)", min_value=0.1, max_value=1000.0, step=0.5, key="mask_target_z")
        col1, col2 = st.columns(2)
        col1.number_input("Start (MHz)", min_value=0.01, max_value=1000.0, step=1.0, key="f_min")
        col2.number_input("End (MHz)", min_value=0.1, max_value=5000.0, step=10.0, key="f_max")
        
    st.divider()
    st.markdown("**【 🎯 Advanced Optimization 】**")
    st.checkbox("Enable Targeted Strike", key="use_target")
    if st.session_state.use_target:
        st.slider("Target Frequency (MHz)", 1.0, 100.0, 15.0, key="target_freq")
        st.slider("Strike Weight", 10, 200, 50, key="target_weight")
    st.divider()

    with st.form("simulation_parameters_form"):
        st.markdown("### 🎛️ Manual Tuning Parameters")
        tab_pcb, tab_pkg, tab_cap = st.tabs(["Stackup & Physical", "Parasitics & PKG", "Decoupling Caps"])
        
        with tab_pcb:
            st.slider("Power Plane Length (mm)", 10.0, 150.0, key="L_mm")
            st.slider("Power Plane Width (mm)", 5.0, 100.0, key="W_mm")
            st.slider("Dielectric Constant (Er)", 1.0, 15.0, key="Er")
            st.slider("Dielectric Thickness d (mil)", 0.1, 20.0, key="d_mil")
            st.slider("Operating Current (A)", 0.1, 100.0, key="I_op")
            
        with tab_pkg:
            st.slider("Capacitor ESL (nH)", 0.01, 2.0, key="ESL")
            st.slider("Capacitor Via Inductance (nH)", 0.0, 1.0, key="L_cap_via")
            st.slider("IC Pin Via Inductance (nH)", 0.0, 1.0, key="L_ic_via")
            st.slider("Package Parasitic Inductance L_pkg (nH)", 0.0, 3.0, key="L_pkg")
            st.slider("On-Die Capacitance C_die (nF)", 0.0, 2000.0, key="C_die")
            
        with tab_cap:
            st.selectbox("Number of Decoupling Capacitors:", [0, 1, 2, 3, 4, 5], key="num_caps")
            def cap_input(label, val_key, unit_key):
                col1, col2 = st.columns([2, 1])
                col1.number_input(label, min_value=0.1, max_value=10000.0, step=0.01, key=val_key)
                col2.selectbox("Unit", ["uF", "nF", "pF"], key=unit_key, label_visibility="collapsed")
                
            cap_input("C1 [Ultra-High Freq]:", "C1_val", "C1_unit")
            cap_input("C2 [High Freq]:", "C2_val", "C2_unit")
            cap_input("C3 [Mid Freq]:", "C3_val", "C3_unit")
            cap_input("C4 [Mid-Low Freq]:", "C4_val", "C4_unit")
            cap_input("C5 [Bulk]:", "C5_val", "C5_unit")
            
        submit_btn = st.form_submit_button("▶️ Update Simulation", type="primary", use_container_width=True)

    st.divider()
    if st.button("🦅 Run Continuous PSO (Auto Tune)", use_container_width=True):
        st.session_state.run_tune = True
        st.rerun()

# ==========================================
# 6. Main UI: Execution & Chart
# ==========================================
if st.session_state.use_band:
    target_z_mohm = st.session_state.mask_target_z
    target_z_ohm = target_z_mohm / 1000.0
else:
    match = re.search(r"(\d+(\.\d+)?)", st.session_state.target_z)
    target_z_mohm = float(match.group(1)) if match else 100.0
    target_z_ohm = target_z_mohm / 1000.0

frequencies = np.logspace(4, 9.7, 300) 

with st.spinner('Calculating PDN profile (Vectorized)...'):
    Z_pcb, Z_die, ir_drop_v = calc_core(
        st.session_state.L_mm * 1e-3, st.session_state.W_mm * 1e-3, st.session_state.Er, 
        st.session_state.d_mil * 25.4e-6, st.session_state.I_op, st.session_state.num_caps, 
        get_cap_val(st.session_state.C1_val, st.session_state.C1_unit),
        get_cap_val(st.session_state.C2_val, st.session_state.C2_unit),
        get_cap_val(st.session_state.C3_val, st.session_state.C3_unit),
        get_cap_val(st.session_state.C4_val, st.session_state.C4_unit),
        get_cap_val(st.session_state.C5_val, st.session_state.C5_unit),
        st.session_state.ESL * 1e-9, st.session_state.L_cap_via * 1e-9, 
        st.session_state.L_ic_via * 1e-9, st.session_state.L_pkg * 1e-9, 
        st.session_state.C_die * 1e-9, frequencies
    )

if st.session_state.use_band:
    f_min_hz = st.session_state.f_min * 1e6
    f_max_hz = st.session_state.f_max * 1e6
    in_band_mask = (frequencies >= f_min_hz) & (frequencies <= f_max_hz)
    max_z_ohms = np.max(Z_die[in_band_mask]) if any(in_band_mask) else np.max(Z_die)
else:
    max_z_ohms = np.max(Z_die)

ac_drop_v = st.session_state.I_op * max_z_ohms
total_drop_v = ir_drop_v + ac_drop_v

st.markdown("### 📊 System Voltage Drop Estimation (Worst-Case Analysis)")
col1, col2, col3 = st.columns(3)

ir_drop_color = "normal" if (ir_drop_v * 1000) <= 50 else "inverse"
col1.metric(label="DC IR Drop", value=f"{ir_drop_v * 1000:.2f} mV", delta="Pass" if ir_drop_color=="normal" else "Warning", delta_color=ir_drop_color)

ac_drop_color = "normal" if max_z_ohms <= target_z_ohm else "inverse"
col2.metric(label="AC Transient Droop (Ripple)", value=f"{ac_drop_v * 1000:.2f} mV", delta="Pass" if ac_drop_color=="normal" else "Violation", delta_color=ac_drop_color)

col3.metric(label="🔥 Estimated Total Voltage Drop", value=f"{total_drop_v * 1000:.2f} mV")

st.divider()

fig = go.Figure()

if st.session_state.use_band:
    f_min_hz = st.session_state.f_min * 1e6
    f_max_hz = st.session_state.f_max * 1e6
    fig.add_vrect(
        x0=f_min_hz, x1=f_max_hz, fillcolor="rgba(0, 255, 0, 0.05)",
        line=dict(color="green", width=1, dash="dash"),
        annotation_text=f"Datasheet Band ({target_z_mohm} mΩ)", annotation_position="top left"
    )
    fig.add_trace(go.Scatter(x=[f_min_hz, f_max_hz], y=[target_z_ohm, target_z_ohm], mode='lines', line=dict(color='red', width=3, dash='solid'), name=f'Band Spec ({target_z_mohm} mΩ)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[frequencies[0], f_min_hz], y=[target_z_ohm, target_z_ohm], mode='lines', line=dict(color='red', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[f_max_hz, frequencies[-1]], y=[target_z_ohm, target_z_ohm], mode='lines', line=dict(color='red', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
else:
    fig.add_trace(go.Scatter(x=[frequencies[0], frequencies[-1]], y=[target_z_ohm, target_z_ohm], mode='lines', line=dict(color='red', width=2, dash='dash'), name=f'Target Spec ({target_z_mohm} mΩ)', hoverinfo='skip'))

if st.session_state.use_target:
    f_target = st.session_state.target_freq * 1e6
    fig.add_vrect(x0=f_target * 0.75, x1=f_target * 1.25, fillcolor="yellow", opacity=0.1, layer="below", line_width=0, annotation_text="Strike Zone", annotation_position="top right")
    fig.add_vline(x=f_target, line_width=2, line_dash="dot", line_color="orange")

fig.add_trace(go.Scatter(x=frequencies, y=Z_pcb, mode='lines', line=dict(color='gray', width=2, dash='dot'), name='Impedance at PCB Pin', hovertemplate='<b>Freq:</b> %{x:.2e} Hz<br><b>Z (PCB):</b> %{y:.4f} Ω<extra></extra>'))
fig.add_trace(go.Scatter(x=frequencies, y=Z_die, mode='lines', line=dict(color='#00BFFF', width=3), name='Impedance at Die Pad', hovertemplate='<b>Freq:</b> %{x:.2e} Hz<br><b>Z (Die):</b> %{y:.4f} Ω<extra></extra>'))

y_min_log = np.log10(min(1e-3, target_z_ohm * 0.3))
y_max_log = np.log10(max(10, max(Z_pcb)*1.2))

fig.update_layout(
    title="Power Delivery Network (Continuous Space Optimization)",
    xaxis_title="Frequency (Hz)", yaxis_title="Impedance (Ohms)",
    xaxis_type="log", yaxis_type="log", yaxis=dict(range=[y_min_log, y_max_log]),
    hovermode="x unified", template="plotly_dark",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0, 0, 0, 0.5)"),
    height=600, margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. BOM Export 
# ==========================================
st.divider()
st.subheader("📋 Optimized BOM List (Continuous Values)")
bom_data = {
    "Location": ["C1 [Ultra-High Freq]", "C2 [High Freq]", "C3 [Mid Freq]", "C4 [Mid-Low Freq]", "C5 [Bulk]"],
    "Continuous Value": [st.session_state.C1_val, st.session_state.C2_val, st.session_state.C3_val, st.session_state.C4_val, st.session_state.C5_val],
    "Unit": [st.session_state.C1_unit, st.session_state.C2_unit, st.session_state.C3_unit, st.session_state.C4_unit, st.session_state.C5_unit],
    "Role": ["Package Resonance Suppression / On-Die", "High-Freq Decoupling", "Mid-Freq Decoupling", "Mid-Low Freq Decoupling", "VRM Bulk Storage"]
}
st.dataframe(pd.DataFrame(bom_data), use_container_width=True, hide_index=True)
st.download_button(label="📥 Download BOM (CSV)", data=pd.DataFrame(bom_data).to_csv(index=False).encode('utf-8'), file_name="pdn_continuous_bom.csv", mime="text/csv", type="secondary")