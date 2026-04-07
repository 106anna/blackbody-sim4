import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="黑體輻射與色彩解碼器", layout="centered")

# --- 物理常數 ---
h, c, kB = 6.626e-34, 3.0e8, 1.38e-23
sigma = 5.67e-8  
T_sun = 5773     
P_sun = sigma * (T_sun**4) 

st.title("🌡️ 黑體輻射：從物理能量到眼睛色彩")
st.markdown("---")

# 1. 數值輸入框
temp_k = st.number_input("請輸入絕對溫度 (Kelvin):", min_value=100, max_value=20000, value=5773, step=100)

# 2. 物理與生物曲線定義
waves_nm = np.linspace(380, 780, 400) # 聚焦可見光區以計算顏色
waves_m = waves_nm * 1e-9

# 普朗克定律函數
def planck(w_m, T):
    with np.errstate(over='ignore', divide='ignore'):
        return (2 * h * c**2) / (w_m**5 * (np.exp((h * c) / (w_m * kB * T)) - 1))

# 錐狀細胞敏感度函數 (精確峰值：S:440, M:545, L:570)
def cone_sensitivity(x, peak, width):
    return np.exp(-0.5 * ((x - peak) / width)**2)

# 3. 計算當前溫度的 S, M, L 響應值 (積分概念)
intensity_vis = planck(waves_m, temp_k)
s_sens = cone_sensitivity(waves_nm, 440, 30)
m_sens = cone_sensitivity(waves_nm, 545, 40)
l_sens = cone_sensitivity(waves_nm, 570, 45)

# 計算相對響應強度 (面積積分)
S_val = np.trapz(intensity_vis * s_sens, waves_nm)
M_val = np.trapz(intensity_vis * m_sens, waves_nm)
L_val = np.trapz(intensity_vis * l_sens, waves_nm)

# 正規化 (以最大值為 1，方便觀察比例)
max_val = max(S_val, M_val, L_val, 1e-10)
S_norm, M_norm, L_norm = S_val/max_val, M_val/max_val, L_val/max_val

# 4. 模擬眼睛看到的顏色 (簡化版 RGB 轉換)
# 將 L, M, S 映射到 R, G, B 顯示
r_display = min(L_norm * 1.2, 1.0) 
g_display = min(M_norm * 1.0, 1.0)
b_display = min(S_norm * 0.8, 1.0)
hex_color = '#%02x%02x%02x' % (int(r_display*255), int(g_display*255), int(b_display*255))

# 5. 數據分析顯示區
st.subheader("🔍 色彩解碼數據")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("S 響應 (藍)", f"{S_norm:.3f}")
with col2:
    st.metric("M 響應 (綠)", f"{M_norm:.3f}")
with col3:
    st.metric("L 響應 (紅)", f"{L_norm:.3f}")

# 顯示模擬顏色
st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: center; background-color: {hex_color}; 
            height: 100px; border-radius: 10px; border: 2px solid #555; color: {'black' if (r_display+g_display+b_display)>1.5 else 'white'}; 
            font-weight: bold; font-size: 24px;">
    模擬眼睛看到的顏色 (T={temp_k}K)
</div>
""", unsafe_allow_html=True)

# 6. 繪製圖表
st.markdown("### 📈 光譜與敏感度重疊圖")
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左軸：能量
ax1.plot(waves_nm, intensity_vis, color='black', lw=3, label='Blackbody Intensity')
ax1.set_ylabel("Radiant Intensity", color='black')

# 右軸：SML 敏感度
ax2 = ax1.twinx()
ax2.plot(waves_nm, l_sens, color='red', ls='--', alpha=0.5)
ax2.plot(waves_nm, m_sens, color='green', ls='--', alpha=0.5)
ax2.plot(waves_nm, s_sens, color='blue', ls='--', alpha=0.5)
ax2.set_ylabel("Relative Sensitivity", color='gray')
ax2.set_ylim(0, 1.3)

# 標籤 (標於 Peak 右側/左側)
ax2.text(570+10, 1.02, 'L', color='red', fontweight='bold', ha='left')
ax2.text(545-10, 1.02, 'M', color='green', fontweight='bold', ha='right')
ax2.text(440+10, 1.02, 'S', color='blue', fontweight='bold', ha='left')

ax1.set_xlabel("Wavelength (nm)")
ax1.grid(True, alpha=0.3)
st.pyplot(fig)

# 7. 不同溫度的對照表
st.markdown("### 📋 典型溫度色彩對照")
st.table([
    {"溫度 (K)": "800 K", "S:M:L 比例": "0.00 : 0.01 : 1.00", "視覺特徵": "暗紅色 (只有紅光剛達標)"},
    {"溫度 (K)": "2773 K", "S:M:L 比例": "0.05 : 0.48 : 1.00", "視覺特徵": "暖黃色 (燈泡色)"},
    {"溫度 (K)": "5773 K", "S:M:L 比例": "0.68 : 0.92 : 1.00", "視覺特徵": "白色 (太陽光/晝光)"},
    {"溫度 (K)": "12000 K", "S:M:L 比例": "1.00 : 0.85 : 0.75", "視覺特徵": "藍白色 (高溫恆星)"},
])
