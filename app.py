import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 網頁基本設定
st.set_page_config(page_title="黑體輻射與色彩", layout="centered")

# --- 物理常數 ---
h, c, kB = 6.626e-34, 3.0e8, 1.38e-23
sigma = 5.67e-8  
T_sun = 5773     

st.title("🌡️ 黑體輻射：
從物理能量到眼睛色彩")
st.markdown("---")

# 1. 數值輸入框
temp_k = st.number_input("請輸入絕對溫度 (Kelvin):", min_value=100, max_value=20000, value=5773, step=100)

# 2. 物理與生物曲線定義
waves_nm = np.linspace(380, 780, 400) # 聚焦可見光區以計算顏色比例
waves_m = waves_nm * 1e-9

def planck(w_m, T):
    with np.errstate(over='ignore', divide='ignore'):
        return (2 * h * c**2) / (w_m**5 * (np.exp((h * c) / (w_m * kB * T)) - 1))

def cone_sensitivity(x, peak, width):
    return np.exp(-0.5 * ((x - peak) / width)**2)

# 計算能量與敏感度
intensity_vis = planck(waves_m, temp_k)
s_sens = cone_sensitivity(waves_nm, 440, 30)
m_sens = cone_sensitivity(waves_nm, 545, 40)
l_sens = cone_sensitivity(waves_nm, 570, 45)

# 🌟 修正後的積分寫法 (相容新舊版本 NumPy)
def get_integral(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

S_val = get_integral(intensity_vis * s_sens, waves_nm)
M_val = get_integral(intensity_vis * m_sens, waves_nm)
L_val = get_integral(intensity_vis * l_sens, waves_nm)

# 正規化比例
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

st.markdown(f"""
<div style="background-color: {hex_color}; height: 80px; border-radius: 10px; 
            display: flex; align-items: center; justify-content: center;
            border: 2px solid #333; color: {'black' if (r+g+b)>1.5 else 'white'}; font-weight: bold; font-size: 20px;">
    模擬黑體輻射與ＳＭＬ錐狀細胞 (T={temp_k}K)
</div>
""", unsafe_allow_html=True)

# 5. 繪製圖表
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Radiant Intensity", color='black')
ax1.plot(waves_nm, intensity_vis, color='black', lw=3, label='Blackbody Spectrum')

peak_nm = (2.898e-3 / temp_k) * 1e9
ax1.axvline(peak_nm, color='darkred', ls=':', lw=2, label=f'Peak: {peak_nm:.1f}nm')

ax2 = ax1.twinx()
ax2.set_ylabel("Relative Sensitivity (0 to 1)", color='gray')
ax2.plot(waves_nm, l_sens, color='red', ls='--', alpha=0.5)
ax2.plot(waves_nm, m_sens, color='green', ls='--', alpha=0.5)
ax2.plot(waves_nm, s_sens, color='blue', ls='--', alpha=0.5)

# 標籤位置：M 在左，S/L 在右
label_y = 1.0
offset_x = 12
ax2.text(570 + offset_x, label_y, 'L', color='red', fontweight='bold', fontsize=14, ha='left', va='center')
ax2.text(545 - offset_x, label_y, 'M', color='green', fontweight='bold', fontsize=14, ha='right', va='center')
ax2.text(440 + offset_x, label_y, 'S', color='blue', fontweight='bold', fontsize=14, ha='left', va='center')

ax2.set_ylim(0, 1.2)
ax1.set_ylim(0, np.max(intensity_vis) * 1.35)
ax1.set_xlim(350, 850)
ax1.legend(loc='upper right', fontsize='small')
ax1.grid(True, alpha=0.3)

st.pyplot(fig)

st.info("""
**💡 數據背後的科學原理：**
* **SML 響應值**：並非直接對應圖上的點，而是**計算了黑色輻射能量曲線與 SML 虛線重疊部分的總面積**（即積分）。這代表了視覺細胞接收到的總刺激能量。
* **色彩判定**：大腦根據 S:M:L 的**面積比例**來解碼色彩。
""")
st.markdown(f"""
### 🎓 思考練習：
1. **為什麼太陽 (5773K) 波峰在綠色，但 L 響應卻是 1.000？
2. **觀察能量偏移**：當溫度調低至燈泡 (2773K) 時，黑色輻射曲線與 **S、M、L** 三條虛線的重疊面積分別發生了什麼變化？這與我們看到的橘黃色有什麼關聯？
""")
