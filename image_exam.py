# image_analyzer.py

import os
from PIL import Image
from colorthief import ColorThief
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import easyocr
from openai import OpenAI
import base64
from dotenv import load_dotenv
import pymongo
from datetime import datetime
import uuid
import cv2
import streamlit as st
import pandas as pd
from pathlib import Path

# Load environment variables
load_dotenv()

class ImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.color_thief = ColorThief(image_path)
        self.image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    def get_palette(self, color_count=12, quality=100):
        """Get color palette using ColorThief"""
        return self.color_thief.get_palette(color_count=color_count, quality=quality)
    
    def filter_duplicate_colors(self, palette):
        """Remove duplicate colors from palette"""
        return list(dict.fromkeys(map(tuple, palette)))
    
    def calculate_color_ratio(self, color1, color2):
        """Calculate WCAG contrast ratio between two colors"""
        def get_luminance(color):
            rgb = np.array(color) / 255.0
            rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        
        l1 = get_luminance(color1)
        l2 = get_luminance(color2)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

    def get_region_colors(self, bbox):
        """Extract colors from a specific region of the image"""
        try:
            x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
            
            height, width = self.image_array.shape[:2]
            x_min = max(0, min(x_min, width - 1))
            x_max = max(0, min(x_max, width))
            y_min = max(0, min(y_min, height - 1))
            y_max = max(0, min(y_max, height))
            
            region = self.image_array[y_min:y_max, x_min:x_max]
            
            if region.size == 0 or region.shape[0] == 0 or region.shape[1] == 0:
                return None, None
            
            pixels = region.reshape(-1, 3)
            
            if len(pixels) < 3:
                return None, None
            
            kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
            colors = kmeans.cluster_centers_
            
            brightness = np.mean(colors, axis=1)
            sorted_indices = np.argsort(brightness)
            
            text_color = tuple(map(int, colors[sorted_indices[0]]))
            bg_color = tuple(map(int, colors[sorted_indices[-1]]))
            
            return text_color, bg_color
            
        except Exception as e:
            st.error(f"Error in get_region_colors: {str(e)}")
            return None, None

    def find_adjacent_colors(self, colors):
        """Find adjacent colors using K-means clustering"""
        if len(colors) < 2:
            return [], []
            
        colors_array = np.array(colors)
        n_clusters = min(len(colors), 5)
        
        kmeans = KMeans(n_clusters=n_clusters).fit(colors_array)
        clustered_colors = kmeans.cluster_centers_
        
        adjacent_colors = []
        distances_list = []
        
        for color in clustered_colors:
            distances = cdist([color], clustered_colors)[0]
            sorted_indices = np.argsort(distances)[1:5]
            adjacent_colors.append(clustered_colors[sorted_indices])
            distances_list.append(distances[sorted_indices])
        
        return adjacent_colors, distances_list

    def analyze_text_color_contrast(self, text_results):
        """Analyze color contrast for each text region"""
        text_contrast_results = []
        
        for text_info in text_results:
            try:
                bbox = text_info[0]
                text_content = text_info[1]
                confidence = float(text_info[2])
                
                points = np.array(bbox)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                text_color, bg_color = self.get_region_colors((x_min, y_min, x_max, y_max))
                
                if text_color is None or bg_color is None:
                    continue
                
                contrast_ratio = float(self.calculate_color_ratio(text_color, bg_color))
                
                wcag_compliance = {
                    'AA_large': contrast_ratio >= 3.0,
                    'AA_small': contrast_ratio >= 4.5,
                    'AAA_large': contrast_ratio >= 4.5,
                    'AAA_small': contrast_ratio >= 7.0
                }
                
                text_contrast_results.append({
                    'text': text_content,
                    'confidence': confidence,
                    'text_color': text_color,
                    'background_color': bg_color,
                    'contrast_ratio': contrast_ratio,
                    'wcag_compliance': wcag_compliance,
                    'position': {
                        'x_min': int(x_min),
                        'x_max': int(x_max),
                        'y_min': int(y_min),
                        'y_max': int(y_max)
                    }
                })
                
            except Exception as e:
                st.error(f"Error analyzing text region: {str(e)}")
                continue
        
        return text_contrast_results

    def extract_text_easyocr(self):
        """Extract text and analyze color contrast using EasyOCR"""
        try:
            reader = easyocr.Reader(['en'])
            results = reader.readtext(self.image_path)
            
            if not results:
                return {'full_text': "", 'text_regions': []}
            
            text_analysis = self.analyze_text_color_contrast(results)
            
            return {
                'full_text': "\n".join([result['text'] for result in text_analysis]),
                'text_regions': text_analysis
            }
        except Exception as e:
            st.error(f"Error extracting text with EasyOCR: {str(e)}")
            return {'full_text': "", 'text_regions': []}

def classify_image(image_path):
    """Classify image using OpenAI API"""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        client = OpenAI(
            api_key=f'{os.environ.get("LLMFOUNDARY_TOKEN")}:my-test-project',
            base_url="https://llmfoundry.straive.com/openai/v1/",
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user",
                 "content":[
                    {"type": "text", 
                     "text": "What type of exact image is this? one word classify the images in any of one classes :'just_image', 'bar_chart', 'diagram', 'flow_chart', 'graph', 'growth_chart', 'pie_chart', 'table','map' 1st classification and also classify Infographic without label, Infographic with label, Map without label, Map with Label,complex images,graphic images 2nd classification give only from both classifcation 1 and 1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
                } 
            ],
            model="gpt-4o-mini",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return "Classification failed"

def analyze_image_accessibility(image_path):
    """Main function to analyze image accessibility"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
        
        analyzer = ImageAnalyzer(image_path)
        
        classification = classify_image(image_path)
        text_results = analyzer.extract_text_easyocr()
        
        palette = analyzer.get_palette(color_count=12, quality=100)
        unique_palette = analyzer.filter_duplicate_colors(palette)
        adjacent_colors, distances_list = analyzer.find_adjacent_colors(unique_palette)
        
        results = {
            'classification': classification,
            'text_analysis': text_results,
            'color_analysis': []
        }
        
        for i in range(len(adjacent_colors)):
            dominant_color = unique_palette[i]
            color_analysis = {
                'dominant_color': dominant_color,
                'adjacent_colors': []
            }
            
            for j in range(len(adjacent_colors[i])):
                adjacent_color = adjacent_colors[i][j]
                distance = float(distances_list[i][j])
                adj_color_tuple = tuple(int(c) for c in adjacent_color)
                contrast_ratio = float(analyzer.calculate_color_ratio(dominant_color, adj_color_tuple))
                
                color_analysis['adjacent_colors'].append({
                    'color': adj_color_tuple,
                    'distance': distance,
                    'contrast_ratio': contrast_ratio
                })
            
            results['color_analysis'].append(color_analysis)
        
        return results
            
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

# app.py

def create_accessibility_app():
    st.set_page_config(
        page_title="Image Accessibility Analyzer",
        page_icon="🖼️",
        layout="wide"
    )

    
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    analysis_depth = st.sidebar.radio(
        "Analysis Detail Level",
        ["Basic", "Detailed"],
        help="Basic shows only compliance status, Detailed shows full analysis"
    )
    
    classification_filter = st.sidebar.multiselect(
        "Filter by Classification",
        ["bar_chart", "diagram", "flow_chart", "graph", "growth_chart", 
         "pie_chart", "table", "map", "infographic", "complex_image"],
        default=[],
        help="Select image types to display"
    )

    # Main content
    

    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        help="Select multiple image files for analysis"
    )

    if uploaded_files:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Analysis container
        analysis_container = st.container()
        
        with analysis_container:
            st.subheader("Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            image_results = []
            
            for idx, file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {file.name} ({idx + 1}/{len(uploaded_files)})")
                
                temp_path = temp_dir / file.name
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                results = analyze_image_accessibility(str(temp_path))
                
                if results:
                    # Determine compliance
                    text_compliant = all(
                        region['wcag_compliance']['AA_small']
                        for region in results['text_analysis']['text_regions']
                    ) if results['text_analysis']['text_regions'] else True
                    color_compliant = all(
                        adj['contrast_ratio'] >= 4.5
                        for color in results['color_analysis']
                        for adj in color['adjacent_colors']
                    )
                    
                    overall_compliance = "Compliant" if text_compliant and color_compliant else "Non-Compliant"
                    compliance_color = "green" if overall_compliance == "Compliant" else "red"
                    
                    image_results.append({
                        'filename': file.name,
                        'classification': results['classification'],
                        'compliance': overall_compliance,
                        'compliance_color': compliance_color,
                        'full_results': results,
                        'image_path': str(temp_path)
                    })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.subheader("Analysis Results")
            
            df = pd.DataFrame(image_results)
            if classification_filter:
                df = df[df['classification'].isin(classification_filter)]
            
            if not df.empty:
                classifications = df['classification'].unique()
                
                for classification in classifications:
                    st.markdown(f"### {classification}")
                    
                    class_images = df[df['classification'] == classification]
                    cols = st.columns(3)
                    
                    for idx, (_, row) in enumerate(class_images.iterrows()):
                        col = cols[idx % 3]
                        
                        with col:
                            # Image display
                            img = Image.open(row['image_path'])
                            st.image(img, caption=row['filename'], use_container_width=True)
                            
                            # Compliance status
                            st.markdown(f"**Status:** :{row['compliance_color']}[{row['compliance']}]")
                            
                            if analysis_depth == "Detailed":
                                with st.expander("View Detailed Analysis"):
                                    results = row['full_results']
                                    
                                    # Text Analysis
                                    if results['text_analysis']['text_regions']:
                                        st.markdown("#### Text Analysis")
                                        for region in results['text_analysis']['text_regions']:
                                            st.markdown(f"- Text: {region['text']}")
                                            st.markdown(f"- Contrast: {region['contrast_ratio']:.2f}")
                                            st.markdown("- WCAG Compliance:")
                                            for level, compliant in region['wcag_compliance'].items():
                                                st.markdown(f"  - {level}: {'✅' if compliant else '❌'}")
                                    
                                    # Color Analysis
                                    st.markdown("#### Color Analysis")
                                    for i, color_data in enumerate(results['color_analysis']):
                                        dominant_rgb = color_data['dominant_color']
                                        st.markdown(f"**Dominant Color {i+1}**")
                                        st.markdown(
                                            f'<div style="background-color:rgb{dominant_rgb};width:50px;height:20px;"></div>',
                                            unsafe_allow_html=True
                                        )
                                        
                                        for j, adj_data in enumerate(color_data['adjacent_colors']):
                                            adj_rgb = adj_data['color']
                                            contrast = adj_data['contrast_ratio']
                                            st.markdown(f"Adjacent Color {j+1}")
                                            st.markdown(
                                                f'<div style="background-color:rgb{adj_rgb};width:50px;height:20px;"></div>',
                                                unsafe_allow_html=True
                                            )
                                            st.markdown(f"Contrast Ratio: {contrast:.2f}")
                                            if contrast >= 7.0:
                                                st.markdown("✅ Meets WCAG AAA")
                                            elif contrast >= 4.5:
                                                st.markdown("✅ Meets WCAG AA")
                                            else:
                                                st.markdown("❌ Does not meet WCAG requirements")
            
            # Summary statistics
            st.markdown("### Summary")
            st.write(f"Total images analyzed: {len(image_results)}")
            compliant_images = df[df['compliance'] == 'Compliant']
            st.write(f"Compliant images: {len(compliant_images)}")
            st.write(f"Non-compliant images: {len(df) - len(compliant_images)}")
   

# Run the application
if __name__ == "__main__":
    create_accessibility_app()
