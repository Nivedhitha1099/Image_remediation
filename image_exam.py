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
import torch
from pathlib import Path
import os
import base64
import easyocr
import cv2
import numpy as np
import re
import multiprocessing
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from rapidfuzz import fuzz
import requests
import shutil
import time
import random
import string
load_dotenv()
# Initialize EasyOCR with multiprocessing
reader = easyocr.Reader(['en'], gpu=False)

# Set up Claude AI
chat_model = ChatAnthropic(
    anthropic_api_key=f'{os.environ.get("LLMFOUNDARY_TOKEN")}:my-test-project',
    anthropic_api_url="https://llmfoundry.straive.com/anthropic/",
    model_name="claude-3-haiku-20240307"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip().lower()
    return text

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_with_easyocr(image_path):
    image = preprocess_image(image_path)
    results = reader.readtext(image)
    
    extracted_data = {}
    for (bbox, text, confidence) in results:
        cleaned_text = clean_text(text)
        if cleaned_text:
            (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
            extracted_data[cleaned_text] = {
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min),
                "height": int(y_max - y_min),
                "confidence": round(confidence, 2),
                "coordinates": bbox
            }
    
    return extracted_data

def call_claude(image_base64, result_dict):
    try:
        message = HumanMessage(content=[
            {"type": "text", "text": 
             """Extract all text elements visible also partially visible in this image. 
             Please:
              1. List all text items exactly as they appear
              2. Include all labels, titles, and captions
              3. Maintain the original phrasing without summarizing
              4. Organize the content in a simple list format with each item on a new line
              5. Do not categorize, interpret, or add contextual information
              6. If text appears unclear or partially visible or even least visible or  faint, or obscured, include it in the list
              7. The answer should be a list of text elements without any headings like "EXTRACTED TEXT:" and "Here is the list..."(do to give any headings like this) and additional information in the start of the list please
              8. Maintain consistent results and accuracy and format across attempts 
    
              - List only text elements here with numbers
              maintain same format across attempts """},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}
        ])
        response = chat_model.invoke([message])
        result_dict["claude_text"] = response.content
    except Exception as e:
        result_dict["claude_text"] = ""
        print(f"Claude API failed: {e}")

def call_easyocr(image_path, result_dict):
    result_dict["easyocr_data"] = extract_with_easyocr(image_path)

def match_texts(claude_texts, easyocr_data):
    matched_texts = []
    unmatched_texts = []

    for phrase in claude_texts:
        best_match, best_score = None, 0
        for easy_text, data in easyocr_data.items():
            score = fuzz.ratio(phrase.lower(), easy_text.lower())
            if score > best_score:
                best_match, best_score = easy_text, score
        
        if best_match and best_score >= 0:
            matched_texts.append({
                "text": phrase,
                "matched_with": best_match,
                "coordinates":easyocr_data[best_match]["coordinates"], 
                "confidence": easyocr_data[best_match]["confidence"]
            })
        else:
            unmatched_texts.append({
                "text": phrase,
                "closest_match": best_match,
                "closest_score": best_score,
                "coordinates": easyocr_data[best_match]["coordinates"],
                "confidence": easyocr_data[best_match]["confidence"]
                    
            })
    
    return matched_texts, unmatched_texts
def get_dominant_color(image_region):
    
    pixels = image_region.reshape(-1, 3)
    
    
    hist_bins = 8
    histograms = []
    
    
    for channel in range(3):
        hist = np.histogram(pixels[:, channel], bins=hist_bins, range=(0, 256))[0]
        histograms.append(hist)
    
    
    max_count = 0
    dominant_color = None
    
    for r_bin in range(hist_bins):
        for g_bin in range(hist_bins):
            for b_bin in range(hist_bins):
                
                count = min(histograms[0][r_bin], histograms[1][g_bin], histograms[2][b_bin])
                
                if count > max_count:
                    max_count = count
                    
                    bin_size = 256 // hist_bins
                    dominant_color = (
                        r_bin * bin_size + bin_size // 2,
                        g_bin * bin_size + bin_size // 2,
                        b_bin * bin_size + bin_size // 2
                    )
    
    return dominant_color


load_dotenv()

class ImageAnalyzer:
    def __init__(self, image_path:str):
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
        
        def get_luminance(color):
            rgb = np.array(color) / 255.0
            rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        
        l1 = get_luminance(color1)
        l2 = get_luminance(color2)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
    def calculate_wcag_contrast_ratio(self, color1, color2):
        

        def get_relative_luminance(color):
            r, g, b = [x / 255.0 for x in color]
            r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.03928 else r / 12.92
            g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.03928 else g / 12.92
            b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.03928 else b / 12.92
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        l1 = get_relative_luminance(color1)
        l2 = get_relative_luminance(color2)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05) if min(l1, l2) > 0 else 0

    def get_region_colors(self, bbox):
        try:
            if len(bbox) != 4:
                 raise ValueError("Bounding box must contain exactly four coordinates.")
            
            debug_dir = "image_remediation_debug"
            os.makedirs(debug_dir, exist_ok=True)
        
            points = np.array(bbox)
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.image_array.shape[1] - 1, x_max)
            y_max = min(self.image_array.shape[0] - 1, y_max)
        
            if x_max <= x_min or y_max <= y_min or (x_max - x_min) < 2 or (y_max - y_min) < 2:
                 return None, None
            
            padding = max(2, int(min(x_max - x_min, y_max - y_min) * 0.05))
            x_min_pad = max(0, x_min - padding)
            y_min_pad = max(0, y_min - padding)
            x_max_pad = min(self.image_array.shape[1] - 1, x_max + padding)
            y_max_pad = min(self.image_array.shape[0] - 1, y_max + padding)
            
            region = self.image_array[y_min:y_max, x_min:x_max].copy()
            region_with_padding = self.image_array[y_min_pad:y_max_pad, x_min_pad:x_max_pad].copy()
               
            
               
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)    
            _, otsu_binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            
            text_mask = otsu_binary == 255
            text_mask = cv2.resize(otsu_binary, (region.shape[1], region.shape[0])) == 255
        
            if np.sum(text_mask) >= 10:
                
                if text_mask.shape[:2] == region.shape[:2]:
                      text_pixels = region[text_mask]
                      actual_text_color = tuple(map(int, np.median(text_pixels, axis=0)))
                else:
                
                     pixels = region.reshape(-1, 3)
                     kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                     kmeans.fit(pixels)
                     colors = kmeans.cluster_centers_
                     brightness = np.sum(colors, axis=1)
                     text_color_idx = np.argmin(brightness)
                     actual_text_color = tuple(map(int, colors[text_color_idx]))
            else:
                pixels = region.reshape(-1, 3)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(pixels)
            
            
                colors = kmeans.cluster_centers_
                brightness = np.sum(colors, axis=1)
            
            
                text_color_idx = np.argmin(brightness)
                actual_text_color = tuple(map(int, colors[text_color_idx]))
            
            text_brightness = sum(actual_text_color) / 3
            if text_brightness < 128:
                 text_color = (0, 0, 0)  
            else:
                 text_color = (255, 255, 255)
            
            bg_mask = ~text_mask
            pixels = region_with_padding.reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
        
        
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
        
        
            sorted_indices = np.argsort(-label_counts)
            sorted_colors = [colors[i] for i in sorted_indices]
        
        
            bg_color = None
            for color in sorted_colors:
                color_tuple = tuple(map(int, color))
                distance = np.linalg.norm(np.array(actual_text_color) - np.array(color_tuple))
                if distance > 250:
                     bg_color = color_tuple
                     print(f"Background color: {bg_color}, distance from text: {distance}")
                     break
        
        
            if bg_color is None:
                 bg_color = tuple(map(int, sorted_colors[0]))

            
            

        
            return text_color, bg_color
            
        
        except Exception as e:
             print(f"Error in get_region_colors: {str(e)}")
             return None,None
    def find_adjacent_colors(self, colors):
        """Find adjacent colors using K-means clustering"""
        if len(colors) < 2:
            return [], []
            
        colors_array = np.array(colors)
        n_clusters = 5
        
        kmeans = KMeans(n_clusters=n_clusters).fit(colors_array)
        clustered_colors = kmeans.cluster_centers_
        if len(clustered_colors) < 2:

            return [], []
        dominant_color = max(set(tuple(color) for color in clustered_colors), key=lambda x: sum(1 for color in clustered_colors if np.array_equal(color, x)))

    
        clustered_colors = [color for color in clustered_colors if np.linalg.norm(np.array(color) - np.array(dominant_color)) >= 10]

    
        unique_colors = []
        for color in clustered_colors:
             if all(np.linalg.norm(np.array(color) - np.array(unique_color)) >= 10 for unique_color in unique_colors):
                 unique_colors.append(color)

        adjacent_colors = []
        distances_list = []

        for idx, color in enumerate(unique_colors):
             distances = cdist([color], unique_colors)[0]
             sorted_indices = np.argsort(distances)[1:len(distances)]

        
             filtered_indices = [
             i for i in sorted_indices
            if distances[i] >= 50 and self.calculate_wcag_contrast_ratio(color, unique_colors[i]) >= 1.5
            ]

             adjacent_colors.append([unique_colors[i] for i in filtered_indices])
             distances_list.append(distances[filtered_indices])

        return adjacent_colors, distances_list

    def analyze_text_color_contrast(self, text_regions):
        
        text_contrast_results = []
        
        for text_info in text_regions:
            try:
                bbox = text_info['coordinates']
                text_content = text_info['text']
                confidence = float(text_info.get('confidence', 0))
                
                points = np.array(bbox)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                text_color, bg_color = self.get_region_colors(bbox)
                
                if text_color is None or bg_color is None:
                    continue
                
                contrast_ratio = float(self.calculate_color_ratio(text_color, bg_color))
                
                wcag_compliance = {
                    'AA_large': contrast_ratio >= 4.5,
                    
                    
                }
               
                text_contrast_results.append({
                    'text': text_content,
                    'confidence': confidence,
                    'text_color': text_color,
                    'background_color': bg_color,
                    'contrast_ratio': contrast_ratio,
                    'wcag_compliance': wcag_compliance,
                    'position': {'x_min': int(x_min),
                        'x_max': int(x_max),
                        'y_min': int(y_min),
                        'y_max': int(y_max)}
                })
                
            except Exception as e:
                st.error(f"Error analyzing text region: {str(e)}")
                continue
        print(f"Text: {text_content}, Contrast Ratio: {contrast_ratio}")
        return text_contrast_results
    def extract_text_easyocr(self):
        
        try:
             image_path = self.image_path
             base64_image = encode_image(image_path)

        
             manager = multiprocessing.Manager()
             result_dict = manager.dict()

             process1 = multiprocessing.Process(target=call_claude, args=(base64_image, result_dict))
             process2 = multiprocessing.Process(target=call_easyocr, args=(image_path, result_dict))

             process1.start()
             process2.start()
             process1.join()
             process2.join()

        
             claude_raw_text = result_dict.get("claude_text", "")
             claude_extracted_texts = [text.strip() for text in claude_raw_text.split("\n") if text.strip()]
             easyocr_data = result_dict.get("easyocr_data", {})

        
             matched_texts, unmatched_texts = match_texts(claude_extracted_texts, easyocr_data)

        
             text_regions = []
             for item in matched_texts + unmatched_texts:
                 region = {
                "text": item["text"],
                "confidence": item.get("confidence", 0),
                "position": {
                    "x_min": item["coordinates"][0] if item["coordinates"] else 0,
                    "y_min": item["coordinates"][1] if item["coordinates"] else 0,
                    "x_max": item["coordinates"][0] + item["coordinates"][2] if item["coordinates"] else 0,
                    "y_max": item["coordinates"][1] + item["coordinates"][3] if item["coordinates"] else 0
                },
                "coordinates": item["coordinates"]  
                }
                 text_regions.append(region)
             contrast_results = self.analyze_text_color_contrast(text_regions)
             print(f"Text: {item['text']}, Contrast Ratio: {contrast_results}")
             return {
            'full_text': "\n".join([item["text"] for item in matched_texts + unmatched_texts]),
            'text_regions': contrast_results
        }
        
        except Exception as e:
             
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
                     "text": """Image Classification Task:

                     Objective: Precisely categorize the input image across two hierarchical classification levels.

                     Level 1 Classification (Select ONE precise category):
                     - just_image
                     - bar_chart
                     - diagram
                     - flow_chart
                     - graph
                     - growth_chart
                     - pie_chart
                     - table
                    - map

                    Level 2 Detailed Classification (Select ONE nuanced subcategory):
                    - Infographic without label
                    - Infographic with label
                    - Map without label
                    - Map with label
                    - Complex image
                    - Graphic image

                    Detailed Classification Guidelines:
                    1. Analyze the image with maximum precision
                    2. Choose the most representative category from each level

                    4. If image characteristics span multiple categories, select the most dominant match
                    5. Ensure classification is based on visual content, structure, and primary purpose

                    Output Format:
                    Level 1 Category: [selected category]
                    Level 2 Category: [selected subcategory]"""
                    },
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

def analyze_image_accessibility(image_path:str):
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
                contrast_ratio = float(analyzer.calculate_wcag_contrast_ratio(dominant_color, adj_color_tuple))
                
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



def create_accessibility_app():
    st.set_page_config(
        page_title="Image Accessibility Analyzer",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("Image Accessibility Analyzer")
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

    
    

    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        help="Select multiple image files for analysis"
    )

    if uploaded_files:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        
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
                
                results = analyze_image_accessibility(temp_path)
                

                if results:
                    
                    text_compliant = all(
                        any(compliant for compliant in region['wcag_compliance'].values())
                        for region in results['text_analysis']['text_regions']
                    ) if results['text_analysis']['text_regions'] else True
                    color_compliant = all(
                        adj['contrast_ratio'] >= 3
                        for color in results['color_analysis']
                        for adj in color['adjacent_colors']
                    )
                    overall_compliance = "Compliant" if text_compliant and color_compliant else "Non-Compliant"
                    text_compliant = "text-compliant" if text_compliant else "Non-compliant"
                    color_compliant = "color-compliant" if color_compliant else "Non-compliant"
                    
                    compliance_color = "green" if overall_compliance == "Compliant" else "red"
                    text_color = "green" if text_compliant == "text-compliant" else "red"
                    color_color = "green" if color_compliant == "color-compliant" else "red"
                    
                    
                    
                    image_results.append({
                        'filename': file.name,
                        'classification': results['classification'],
                        'compliance': overall_compliance,
                        'text_compliance': text_compliant,
                        'color_compliance': color_compliant,
                        'compliance_color': compliance_color,
                        'text_color': text_color,
                        'color_color': color_color,
                        'full_results': results,
                        'image_path': str(temp_path)
                    })
            
            
            progress_bar.empty()
            status_text.empty()
                
            
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
                            
                            img = Image.open(row['image_path'])
                            st.image(img, caption=row['filename'])
                            
                             
                            st.markdown(f"**Status:** :{row['compliance_color']}[{row['compliance']}]")
                            st.markdown(f"**Text Compliance:** :{row['text_color']}[{row['text_compliance']}]")
                            st.markdown(f"**Color Compliance:** :{row['color_color']}[{row['color_compliance']}]")
                            
                            if analysis_depth == "Detailed":
                                with st.expander("View Detailed Analysis"):
                                    results = row['full_results']
                                    
                                    
                                    if 'text_analysis' in results and 'text_regions' in results['text_analysis'] and results['text_analysis']['text_regions']:
                                        st.markdown("#### Text Analysis")
                                        for region in results['text_analysis']['text_regions']:
                                            st.markdown(
                                                f"""
                                                <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                                                    <p><strong>Text:</strong> {region['text']}</p>
                                                    <p><strong>Contrast:</strong> {region['contrast_ratio']:.2f}</p>
                                                    <p><strong>WCAG Compliance:</strong></p>
                                                    {''.join([f'<div style="background-color: {"#90EE90" if compliant else "#FFB6C1"}; padding: 5px; margin: 2px;">{level}: {"✅" if compliant else "❌"}</div>' for level, compliant in region['wcag_compliance'].items()])}
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )
                                    
                                   
                                    st.markdown("Color Analysis")
                                    for i, color_data in enumerate(results['color_analysis']):
                                        dominant_rgb = color_data['dominant_color']
                                        st.markdown(f"""
                                        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                                        <p><strong>Dominant Color {i+1}</strong></p>
                                        <div style="display: flex; align-items: center;">
                                        <div style="background-color:rgb{dominant_rgb};width:50px;height:20px;margin-right:10px;"></div>
                                        <span>RGB{dominant_rgb}</span>
                                        </div>
                                        <div style="display: flex; flex-wrap: wrap; margin-top: 10px;">""",
                                        unsafe_allow_html=True
                                       )
                                        for j, adj_data in enumerate(color_data['adjacent_colors']):
                                            if adj_data['contrast_ratio']>=1.5:
                                                 st.markdown(f"""
                                            <div style="flex: 1 1 45%; min-width: 200px; margin: 5px; padding: 10px; border: 1px solid #eee; border-radius: 5px;">
                                            <p style="margin: 0;"><strong>Adjacent Color {j+1}</strong></p>
                                            <div style="display: flex; align-items: center; margin-top: 5px;">
                                            <div style="background-color:rgb{adj_data['color']};width:30px;height:15px;margin-right:10px;"></div>
                                            <span>RGB{adj_data['color']}</span>
                                            </div>
                                            <p style="margin: 5px 0;">Contrast Ratio: {adj_data['contrast_ratio']:.2f}</p>
                                            <div style="background-color: {'#90EE90' if adj_data['contrast_ratio'] >= 3 else '#FFB6C1'}; padding: 5px; border-radius: 3px; font-size: 0.9em;">
                                            {'✅ Meets WCAG AA' if adj_data['contrast_ratio'] >= 3 else '❌ Does not meet WCAG AA'}
                                            </div>
                                            </div>
                                            """
                                            ,unsafe_allow_html=True)
                                            
                                        

            
            st.markdown("### Summary")
            st.write(f"Total images analyzed: {len(image_results)}")
            compliant_images = df[df['compliance'] == 'Compliant']
            text_bg_compliant_images= df[df['text_compliance']=='text-compliant']
            color_compliant_images= df[df['color_compliance']=='color-compliant']
            st.write(f"Text_compliant_images:{len(text_bg_compliant_images)}")
            st.write(f"Color_compliant_images:{len(color_compliant_images)}")
            st.write(f"Compliant images: {len(compliant_images)}")
            st.write(f"Non-compliant images: {len(df) - len(compliant_images)}")
   

if __name__ == "__main__":
    create_accessibility_app()
