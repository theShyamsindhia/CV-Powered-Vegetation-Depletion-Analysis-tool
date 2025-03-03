import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App
st.title("Vegetation Depletion Analysis")
st.markdown("Analyze deforestation trends, view vegetation coverage, and detect reforestation changes interactively.")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload Satellite Images (Chronological Order)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

# Sidebar Parameters
st.sidebar.header("Analysis Parameters")
top_crop = st.sidebar.slider("Top Crop Threshold", 0, 300, 50)
bottom_crop = st.sidebar.slider("Bottom Crop Threshold", 0, 300, 50)
grvi_threshold = st.sidebar.slider("GRVI Threshold for Vegetation", -1.0, 1.0, 0.2, step=0.01)

if uploaded_files:
    # Load Images
    images = [Image.open(file) for file in uploaded_files]
    years = [f"Year {i+1}" for i in range(len(images))]

    # Image Comparison Section
    if len(images) >= 2:
        st.subheader("Compare Satellite Images")
        image_1 = np.array(images[0])
        image_2 = np.array(images[-1])
        image_comparison(
            img1=image_1,
            img2=image_2,
            label1=f"Image {years[0]}",
            label2=f"Image {years[-1]}",
            width=700,
            starting_position=50,
        )

    # Crop Images
    def crop_image(img, top, bottom):
        img_array = np.array(img)
        height, width, _ = img_array.shape
        return img_array[top:height-bottom, :]

    cropped_images = [crop_image(img, top_crop, bottom_crop) for img in images]

    # Calculate GRVI
    def calculate_grvi(img):
        # Convert to BGR for OpenCV compatibility
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        green = img[:, :, 1].astype(np.float32)
        red = img[:, :, 2].astype(np.float32)
        grvi = (green - red) / (green + red + 1e-5)  # Avoid division by zero
        return np.clip(grvi, -1.0, 1.0)  # Ensure values are within [-1, 1]

    grvi_images = [calculate_grvi(img) for img in cropped_images]

    # Normalize GRVI for Visualization
    def normalize_grvi(grvi):
        norm_grvi = cv2.normalize(grvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_grvi = cv2.applyColorMap(norm_grvi, cv2.COLORMAP_JET)
        # Convert from BGR to RGB for proper display
        return cv2.cvtColor(colored_grvi, cv2.COLOR_BGR2RGB)

    normalized_grvi_images = [normalize_grvi(grvi) for grvi in grvi_images]

    # Display GRVI Visualizations
    st.subheader("Normalized GRVI Images")
    cols = st.columns(len(normalized_grvi_images))
    for i, (col, norm_grvi) in enumerate(zip(cols, normalized_grvi_images)):
        with col:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(norm_grvi)
            st.image(pil_image, caption=f"GRVI Normalized {years[i]}", use_column_width=True)

    # Image Comparison Section
    if len(normalized_grvi_images) >= 2:
        st.subheader("Compare GRVI Analysis")
        image_comparison(
            img1=normalized_grvi_images[0],
            img2=normalized_grvi_images[-1],
            label1=f"GRVI {years[0]}",
            label2=f"GRVI {years[-1]}",
            width=700,
            starting_position=50,
        )

    # Vegetation Coverage Calculation
    def calculate_vegetation_coverage(grvi, threshold):
        return np.sum(grvi > threshold) / grvi.size * 100

    vegetation_coverage = [calculate_vegetation_coverage(grvi, grvi_threshold) for grvi in grvi_images]

    # Vegetation Coverage Graph
    st.subheader("Vegetation Coverage Over Time")
    fig, ax = plt.subplots()
    ax.plot(years, vegetation_coverage, marker="o", color="green")
    ax.set_title("Vegetation Coverage Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Coverage (%)")
    st.pyplot(fig)

    # Deforestation Masks
    st.subheader("Deforestation Masks")
    deforestation_masks = [(grvi < grvi_threshold).astype(np.uint8) for grvi in grvi_images]
    for i, mask in enumerate(deforestation_masks):
        # Convert binary mask to RGB format
        display_mask = np.stack([mask * 255] * 3, axis=-1)
        # Convert to PIL Image for proper display
        mask_image = Image.fromarray(display_mask.astype(np.uint8))
        st.image(mask_image, caption=f"Deforestation Mask {years[i]}", use_column_width=True)

    # Cumulative Deforestation Heatmap
    st.subheader("Cumulative Deforestation Hotspot Map")
    cumulative_deforestation = np.sum(deforestation_masks, axis=0)
    fig, ax = plt.subplots()
    sns.heatmap(cumulative_deforestation, cmap="Reds", ax=ax, cbar_kws={"label": "Deforestation Intensity"})
    ax.set_title("Cumulative Deforestation")
    st.pyplot(fig)

    # Deforestation and Reforestation Overlay
    def calculate_reforestation_overlay(masks):
        overlays = []
        for i in range(1, len(masks)):
            diff = masks[i].astype(int) - masks[i-1].astype(int)
            overlay = np.zeros((*diff.shape, 3), dtype=np.uint8)
            overlay[diff == 1] = [0, 0, 255]  # Blue for reforestation
            overlay[diff == -1] = [255, 0, 0]  # Red for deforestation
            overlays.append(overlay)
        return overlays

    overlays = calculate_reforestation_overlay(deforestation_masks)

    st.subheader("Deforestation and Reforestation Overlays")
    for i, overlay in enumerate(overlays):
        # Convert to PIL Image for proper display
        overlay_image = Image.fromarray(overlay)
        st.image(
            overlay_image,
            caption=f"Changes from {years[i]} to {years[i+1]}", 
            use_column_width=True
        )

    # Yearly Vegetation Change Graph
    st.subheader("Yearly Vegetation Change")
    yearly_changes = [
        {
            'year': years[i+1],
            'absolute_change': vegetation_coverage[i+1] - vegetation_coverage[i],
            'percent_change': ((vegetation_coverage[i+1] - vegetation_coverage[i]) / vegetation_coverage[i] * 100)
            if vegetation_coverage[i] != 0 else 0
        }
        for i in range(len(vegetation_coverage)-1)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Absolute change
    changes = [change['absolute_change'] for change in yearly_changes]
    years_plot = [change['year'] for change in yearly_changes]
    colors = ['red' if change < 0 else 'blue' for change in changes]
    
    ax1.bar(years_plot, changes, color=colors)
    ax1.set_title("Absolute Vegetation Change")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Change in Coverage (%)")
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Percentage change
    percent_changes = [change['percent_change'] for change in yearly_changes]
    colors = ['red' if change < 0 else 'blue' for change in percent_changes]
    
    ax2.bar(years_plot, percent_changes, color=colors)
    ax2.set_title("Relative Vegetation Change")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Relative Change (%)")
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Update Summary section with more detailed statistics
    st.subheader("Summary of Findings")
    max_loss = min(percent_changes) if percent_changes else 0
    max_gain = max(percent_changes) if percent_changes else 0
    
    st.markdown(f"""
    - Total Deforestation Detected: {np.sum(cumulative_deforestation)}
    - Initial Vegetation Coverage: {vegetation_coverage[0]:.2f}%
    - Final Vegetation Coverage: {vegetation_coverage[-1]:.2f}%
    - Largest Vegetation Loss: {max_loss:.2f}%
    - Largest Vegetation Gain: {max_gain:.2f}%
    - Average Yearly Change: {np.mean(percent_changes):.2f}%
    """)
else:
    st.info("Please upload at least two satellite images to analyze vegetation changes.")

