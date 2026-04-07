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

# 2. 物理計算與 SML 響應
waves_nm = np.linspace(380, 780, 400) # 聚焦可見光區計算顏色
waves_m = waves_nm * 1e-9

def planck(w_m, T):
    with np.errstate(over='ignore', divide='ignore'):
        return (2 * h * c**2) / (w_m**5 * (np.exp((h * c) / (w_m * kB * T)) - 1))

def cone_sensitivity(x, peak, width):
    return np.exp(-0.5 * ((x - peak) / width)**2)

# 計算積分響應 (S, M, L)
intensity_vis = planck(waves_m, temp_k)
s_sens = cone_sensitivity(waves_nm, 440, 30)
m_sens = cone_sensitivity(waves_nm, 545, 40)
l_sens = cone_sensitivity(waves_nm, 570, 45)

S_val = np.trapz(intensity_vis * s_sens, waves_nm)
M_val = np.trapz(intensity_vis * m_sens, waves_nm)
L_val = np.trapz(intensity_vis * l_sens, waves_nm)

# 正規化比例 (Relative Sensitivity)
max_val = max(S_val, M_val, L_val, 1e-10)
S_norm, M_norm, L_norm = S_val/max_val, M_val/max_val, L_val/max_val

# 3. 顏色模擬 (RGB 映射)
r = min(L_norm * 1.1, 1.0)
g = min(M_norm * 1.0, 1.0)
b = min(S_norm * 0.9, 1.0)
hex_color = '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

# 4. 數據分析顯示
st.subheader("🔍 色彩解碼數據")
c1, c2, c3 = st.columns(3)
c1.metric("S 響應 (藍)", f"{S_norm:.3f}")
c2.metric("M 響應 (綠)", f"{M_norm:.3f}")
c3.metric("L 響應 (紅)", f"{L_norm:.3f}")

# 顯示模擬顏色方塊
st.markdown(f"""
<div style="background-color: {hex_color}; height: 80px; border-radius: 10px; 
            display: flex; align-items: center; justify-content: center;
            border: 2px solid #333; color: {'black' if (r+g+b)>1.5 else 'white'}; font-weight: bold;">
    模擬眼睛看到的顏色 (T={temp_k}K)
</div>
""", unsafe_allow_html=True)

# 5. 繪製圖表 (雙 Y 軸)
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左軸：物理強度
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Radiant Intensity", color='black')
ax1.plot(waves_nm, intensity_vis, color='black', lw=3, label='Blackbody Spectrum')

# 垂直虛線指引波峰
peak_nm = (2.898e-3 / temp_k) * 1e9
ax1.axvline(peak_nm, color='darkred', ls=':', lw=2, label=f'Peak: {peak_nm:.1f}nm')

# 右軸：SML 敏感度
ax2 = ax1.twinx()
ax2.set_ylabel("Relative Sensitivity (0 to 1)", color='gray')
ax2.plot(waves_nm, l_sens, color='red', ls='--', alpha=0.5)
ax2.plot(waves_nm, m_sens, color='green', ls='--', alpha=0.5)
ax2.plot(waves_nm, s_sens, color='blue', ls='--', alpha=0.5)

# 標籤位置：M 在左，S/L 在右
label_y = 1.02
ax2.text(570 + 10, label_y, 'L', color='red', fontweight='bold', ha='left')
ax2.text(545 - 10, label_y, 'M', color='green', fontweight='bold', ha='right')
ax2.text(440 + 10, label_y, 'S', color='blue', fontweight='bold', ha='left')

ax2.set_ylim(0, 1.2)
ax1.set_ylim(0, np.max(intensity_vis) * 1.3)
ax1.legend(loc='upper right', fontsize='small')
st.pyplot(fig)

st.info("💡 **教學重點：** 當 S:M:L 比例接近（如太陽 5773K）時，大腦解讀為白色。當 L 遠大於 S 時，則呈現暖黃或紅色。")
