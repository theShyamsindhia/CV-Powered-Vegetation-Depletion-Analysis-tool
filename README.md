# 🌍 AI-Powered Vegetation Depletion Analysis  
### **Detect. Analyze. Predict.**  
📡 **Satellite Image Processing | AI-Powered Deforestation Tracking | Interactive Data Visualization**  

---

## 🔥 Why This Matters  
Deforestation and vegetation depletion are **urgent global concerns**. With satellite imagery and AI-driven analysis, we can **quantify environmental changes, predict trends, and drive data-backed solutions.**  

This project brings together **computer vision, AI, and geospatial analysis** to track vegetation loss and detect reforestation changes using satellite images.  

✅ **Compare multi-year satellite images** to measure vegetation depletion.  
✅ **Apply GRVI (Green-Red Vegetation Index) for real-time analysis.**  
✅ **Generate deforestation heatmaps & vegetation coverage trends.**  
✅ **Get interactive visualizations & AI-powered insights.**  

---

## 🎯 How It Works  

### **🛰️ 1. Upload Satellite Images**  
- Load **before-and-after** images of forests or vegetation areas.  
- The AI system processes the images & extracts meaningful vegetation data.  

### **🌱 2. Compute Green-Red Vegetation Index (GRVI)**  
- Uses **GRVI** to analyze vegetation health in each image.  
- The formula is:  
  \[
  GRVI = \frac{G - R}{G + R}
  \]
  *(G = Green Channel, R = Red Channel)*  

### **🌍 3. Track Vegetation Loss & Reforestation**  
- AI-driven **heatmaps** highlight changes in forest density.  
- **Deforestation masks** show areas of loss.  
- **Reforestation overlays** detect vegetation recovery.  

### **📊 4. View Vegetation Trends Over Time**  
- Generates **time-series graphs** for long-term analysis.  
- **Customizable threshold sliders** let users fine-tune vegetation classification.  

---

## 🚀 Key Features  

🌿 **Tracks Vegetation Depletion Over Time** → Compare satellite images from different years.  
📊 **Calculates Green-Red Vegetation Index (GRVI)** → Detects vegetation coverage changes using AI-based analysis.  
🌎 **Generates Heatmaps & Deforestation Masks** → Highlights areas of significant change.  
🔄 **Identifies Reforestation Trends** → Detects areas where vegetation is regrowing.  
🎨 **Visualizes Results with Interactive Graphs & Overlays** → No complex GIS software required!  

### **👁️ Interactive Features**  
- Upload satellite images **chronologically** to see changes.  
- Compare **before & after images** with **interactive sliders**.  
- Adjust **vegetation detection thresholds** for **customized** analysis.  
- Generate **dynamic deforestation overlays**.  
- View **time-series vegetation change trends** using AI-driven insights.  

---

## 🛠️ Technologies Used  

This project combines **AI-driven analysis** with real-time data visualization:  

🖥️ **Computer Vision** → OpenCV for image processing & vegetation tracking.  
📡 **Remote Sensing Techniques** → GRVI (Green-Red Vegetation Index) for detecting forest health.  
🧠 **Machine Learning & AI** → AI-based deforestation hotspot analysis.  
📊 **Interactive Data Visualization** → Streamlit + Matplotlib + Seaborn.  

### **Key Libraries**  
- `streamlit` → Interactive web-based visualization  
- `opencv-python` → Image processing  
- `numpy` → Fast array computations  
- `matplotlib & seaborn` → Graphs & heatmaps  
- `PIL` → Image handling  
- `streamlit_image_comparison` → Before-after image sliders  

---

## 📦 Installation & Setup  

### **1️⃣ Install Dependencies**
```bash
pip install streamlit numpy opencv-python pillow matplotlib seaborn streamlit_image_comparison
