import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import cv2
import math
import os
# Added imports for handling image uploads
import base64
from io import BytesIO
from PIL import Image

# --- 1. Global Initialization for Dash and Gunicorn ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server

# --- 2. Data and Helper Functions ---

def get_local_images():
    """Scans the current directory for .jpg and .jpeg files and returns a list of filenames."""
    image_files = []
    try:
        for filename in os.listdir('.'):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                image_files.append(filename)
    except Exception as e:
        print(f"Error scanning directory for images: {e}")
    return image_files

def load_image_from_file(filename):
    """Loads a local image file (JPG/JPEG) into an RGB numpy array using OpenCV."""
    if not filename:
        return None
    
    try:
        # Read the image using OpenCV (loads as BGR)
        img_bgr = cv2.imread(filename)
        if img_bgr is None:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
        
    except Exception as e:
        print(f"Error loading local image '{filename}': {e}")
        return None

def parse_uploaded_contents(contents):
    """Decodes the uploaded base64 image content into an RGB numpy array using PIL."""
    if contents is None:
        return None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Using PIL to handle various image formats correctly
        img = Image.open(BytesIO(decoded))
        # Ensure the image is converted to RGB format
        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return None

def process_image(img_rgb, threshold, threshold_type_str):
    """
    Processes the entire image based on threshold and spot type, 
    finds spots, marks them, and calculates density statistics.
    """
    if img_rgb is None:
        return go.Figure(), 0, 0, 0, 0.0

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_rgb.shape[:2]
    total_area = H * W
    
    # --- 2.1 Convert to Greyscale ---
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_bgr_copy = img_bgr.copy()
    
    # --- Determine Thresholding Mode ---
    if threshold_type_str == 'bright_spots':
        # THRESH_BINARY: Pixels > threshold become white (255) -> Detects bright spots
        t_type = cv2.THRESH_BINARY
    else:
        # THRESH_BINARY_INV: Pixels <= threshold become white (255) -> Detects dark spots
        t_type = cv2.THRESH_BINARY_INV
        
    # --- Binarize Image (Step 3) ---
    _, binary_img = cv2.threshold(img_gray, threshold, 255, t_type)
    
    # Calculate white pixel area
    total_white_area = np.sum(binary_img == 255)
    
    # Calculate ratio
    ratio_white_to_total = total_white_area / total_area if total_area > 0 else 0.0
    
    # --- Find Contours (Spots) ---
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to exclude small noise (e.g., minimum area of 5 pixels)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
    num_spots = len(valid_contours)
    
    # --- Draw Contours (Step 4: Green marks on analyzed image) ---
    cv2.drawContours(img_bgr_copy, valid_contours, -1, (0, 255, 0), 2) # Green contours
    
    # Convert back to RGB for Plotly figure
    img_rgb_marked = cv2.cvtColor(img_bgr_copy, cv2.COLOR_BGR2RGB)
    
    # --- Create Processed Figure ---
    fig_processed = go.Figure(go.Image(z=img_rgb_marked))
    fig_processed.update_layout(
        title=f"Analyzed Image (W:{W} x H:{H}) - Spots Counted: {num_spots}",
        xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_range=[0, W],
        yaxis_range=[H, 0], # Invert Y axis for standard image display
    )
    fig_processed.update_xaxes(showgrid=False, zeroline=False, showticklabels=True)
    fig_processed.update_yaxes(showgrid=False, zeroline=False, showticklabels=True)

    return fig_processed, num_spots, total_area, total_white_area, ratio_white_to_total


# --- 3. Dash Layout ---

# Scan for images once at startup to populate the initial dropdown
available_images = get_local_images()
initial_value = available_images[0] if available_images else None

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Full Image Spot Density Analyzer", className="text-center text-primary my-4"), width=12)
    ]),

    # --- File Input Row (Dropdown AND Upload) ---
    dbc.Row([
        # LEFT: Dropdown for local files
        dbc.Col([
            html.H5("Select Local JPG/JPEG File:", className="mt-3"),
            dcc.Dropdown(
                id='image-dropdown',
                options=[{'label': f, 'value': f} for f in available_images],
                placeholder='Select a file from the folder...',
                value=initial_value,
                style={'width': '100%'}
            ),
        ], width=4),

        # MIDDLE: Upload component for any file
        dbc.Col([
            html.H5("OR Upload Any Image File:", className="mt-3"),
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%', 'height': '40px', 'lineHeight': '40px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center'
                },
                multiple=False
            ),
        ], width=4),

        dbc.Col(html.Div(id='file-status-message', className="font-weight-bold pt-2"), width=4)
    ], className="mb-4 align-items-center"),
    
    # Hidden storage for uploaded image data (base64 string)
    html.Div(id='uploaded-image-data', style={'display': 'none'}),


    # --- Controls Row: Threshold Type and Value ---
    dbc.Row([
        # Threshold Type Selector
        dbc.Col(html.H5("Spot Detection Type:", className="mt-3"), width=4),
        dbc.Col(dcc.RadioItems(
            id='threshold-type',
            options=[
                {'label': ' Bright Spots (on Dark Background) -> White Binary', 'value': 'bright_spots'},
                {'label': ' Dark Spots (on Bright Background) -> White Binary', 'value': 'dark_spots'}
            ],
            value='bright_spots',
            inline=True,
            className="mt-3"
        ), width=8),

        # Threshold Value Slider
        dbc.Col(html.H5("Threshold Value (0-255):", className="mt-3"), width=12),
        dbc.Col(dcc.Slider(
            id='threshold-slider',
            min=0, max=255, step=1, value=127,
            marks={i: str(i) for i in range(0, 256, 32)},
        ), width=10),
        dbc.Col(html.Div(id='threshold-output', className="font-weight-bold pt-2"), width=2)
    ], className="mb-4 align-items-center"),

    # --- Main Display and Stats Row ---
    dbc.Row([
        # LEFT: Analysis Stats
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Analysis Results (Full Image)", className="card-title text-success"),
                    html.Div(id='stats-num-spots', className="h5"),
                    html.Div(id='stats-total-area'),
                    html.Div(id='stats-white-area'),
                    html.Div(id='stats-ratio', className="h4 text-danger mt-2"),
                ]),
                className="mb-4 shadow-sm"
            )
        ], width=4),
        
        # RIGHT: Processed Image Figure
        dbc.Col(
            dcc.Graph(id='processed-image-graph', style={'height': '70vh', 'min-height': '500px'}),
            width=8
        )
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.P("Instructions: Select a local file from the dropdown OR upload a file. Use the threshold controls to correctly isolate the features. The green outlines indicate detected spots.", className="text-center text-muted"), width=12)
    ])

], fluid=True)


# --- 4. Callbacks ---

# Callback A: Store uploaded file contents in a hidden div
@app.callback(
    Output('uploaded-image-data', 'children'),
    Input('upload-image', 'contents'),
)
def store_uploaded_image(contents):
    """Stores the base64 string of the uploaded image."""
    if contents:
        # Stores the raw base64 string provided by dcc.Upload
        return contents
    return None

# Callback B: Update Threshold Display
@app.callback(
    Output('threshold-output', 'children'),
    Input('threshold-slider', 'value')
)
def update_threshold_output(value):
    return f"{value}"

# Callback C: Main Analysis Logic (Reacts to Upload or Dropdown)
@app.callback(
    Output('processed-image-graph', 'figure'),
    Output('stats-num-spots', 'children'),
    Output('stats-total-area', 'children'),
    Output('stats-white-area', 'children'),
    Output('stats-ratio', 'children'),
    Output('file-status-message', 'children'),
    
    # Inputs:
    Input('image-dropdown', 'value'),         # For local files
    Input('uploaded-image-data', 'children'), # For uploaded files (takes priority)
    Input('threshold-slider', 'value'),
    Input('threshold-type', 'value'),
)
def update_analysis(filename_dropdown, uploaded_base64_data, threshold, threshold_type):
    
    img_rgb = None
    source_name = "None"
    
    # --- 1. Determine Source and Load Image ---
    if uploaded_base64_data:
        # Source 1: Uploaded image (highest priority)
        img_rgb = parse_uploaded_contents(uploaded_base64_data)
        source_name = "Uploaded Image"
    elif filename_dropdown:
        # Source 2: Local file selected via dropdown
        img_rgb = load_image_from_file(filename_dropdown)
        source_name = f"Local File: {filename_dropdown}"
    
    # --- 2. Handle No Image or Load Failure ---
    if img_rgb is None:
        default_fig = go.Figure().update_layout(title="No image loaded. Select a file or upload one.", height=500)
        status_msg = html.Div(f"No image available from source: {source_name}. Please load a valid file.", className="text-warning")
        return (default_fig, html.H5("Spots Detected: 0"), 
                html.Div("Total Area: 0 pixels"), html.Div("Thresholded Area: 0 pixels"), 
                html.Div("Ratio (White/Total): 0.0000%"), status_msg)

    # Status update for successful load
    status_msg = html.Div(f"Image loaded successfully from: {source_name}", className="text-success")

    # --- 3. Process and Get Results ---
    fig_processed, num_spots, total_area, total_white_area, ratio_white_to_total = \
        process_image(img_rgb, threshold, threshold_type)

    # --- 4. Format Stats Output ---
    stats_num_spots = html.H5(f"Spots Detected: {num_spots}")
    stats_total_area = html.Div(f"Total Image Area: {total_area:,} pixels")
    stats_white_area = html.Div(f"Thresholded White Area: {total_white_area:,} pixels")
    stats_ratio = html.Div(f"Ratio (White/Total): {ratio_white_to_total*100:.4f}%")
    
    return fig_processed, stats_num_spots, stats_total_area, stats_white_area, stats_ratio, status_msg

# Standard practice to run the app locally if executed directly
if __name__ == '__main__':
    app.run(debug=True)
