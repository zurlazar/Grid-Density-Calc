import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import cv2
import math
import os

# --- 1. Global Initialization for Dash and Gunicorn ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server

# --- 2. Data and Helper Functions ---

def load_image_from_file(filename):
    """
    Loads an image directly from the local filesystem (must be in the same 
    folder as app.py) into an RGB numpy array.
    Returns RGB numpy array or None on failure.
    """
    if not filename:
        return None
    
    # In a local environment, this looks for the file in the script's directory.
    # In the canvas environment, it looks relative to the execution context.
    try:
        # Read the image using OpenCV (loads as BGR)
        img_bgr = cv2.imread(filename)
        if img_bgr is None:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
        
    except Exception as e:
        print(f"Error loading image '{filename}': {e}")
        return None

def process_image(img_rgb, threshold, threshold_type_str):
    """
    Processes the entire image (no ROI) based on threshold and spot type, 
    finds spots, marks them, and calculates density statistics.
    
    Args:
        img_rgb (np.array): The full original image array (RGB).
        threshold (int): The binary threshold value (0-255).
        threshold_type_str (str): 'bright_spots' or 'dark_spots'
        
    Returns:
        tuple: (processed_figure, num_spots, total_area, total_white_area, ratio_white_to_total)
    """
    if img_rgb is None:
        return go.Figure(), 0, 0, 0, 0.0

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_rgb.shape[:2]
    total_area = H * W
    
    # --- 2.1 Convert to Greyscale (Step 2) ---
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

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Full Image Spot Density Analyzer", className="text-center text-primary my-4"), width=12)
    ]),

    # --- File Input Row ---
    dbc.Row([
        dbc.Col(html.H5("Image Filename (must be in the same folder):", className="mt-3"), width=3),
        dbc.Col(dcc.Input(
            id='filename-input',
            type='text',
            placeholder='e.g., my_grid.jpg',
            value='my_image.jpg', # Example placeholder value
            style={'width': '100%'}
        ), width=5),
        dbc.Col(html.Div(id='file-status-message', className="font-weight-bold pt-2"), width=4)
    ], className="mb-4 align-items-center"),

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
        dbc.Col(html.P("Instructions: 1. Ensure your image file (e.g., 'grid.jpg') is in the same folder as this script. 2. Enter the filename above. 3. Select Bright/Dark spots. 4. Adjust the threshold slider to correctly isolate the features (which should turn white in the binary step). 5. The analyzed image will show green outlines on detected spots.", className="text-center text-muted"), width=12)
    ])

], fluid=True)


# --- 4. Callbacks ---

# Callback 1: Update Threshold Display
@app.callback(
    Output('threshold-output', 'children'),
    Input('threshold-slider', 'value')
)
def update_threshold_output(value):
    return f"{value}"

# Callback 2: Main Analysis Logic
@app.callback(
    Output('processed-image-graph', 'figure'),
    Output('stats-num-spots', 'children'),
    Output('stats-total-area', 'children'),
    Output('stats-white-area', 'children'),
    Output('stats-ratio', 'children'),
    Output('file-status-message', 'children'),
    
    Input('filename-input', 'value'),
    Input('threshold-slider', 'value'),
    Input('threshold-type', 'value'),
)
def update_analysis(filename, threshold, threshold_type):
    
    # 1. Load Image
    img_rgb = load_image_from_file(filename)
    
    # Handle image load failure
    if img_rgb is None:
        default_fig = go.Figure().update_layout(title=f"Awaiting image file: {filename}", height=500)
        status_msg = html.Div(f"Error: File '{filename}' not found or could not be loaded. Check filename and folder.", className="text-danger")
        return (default_fig, html.H5("Spots Detected: 0"), 
                html.Div("Total Area: 0 pixels"), html.Div("Thresholded Area: 0 pixels"), 
                html.Div("Ratio (White/Total): 0.0000%"), status_msg)

    # Status update for successful load
    status_msg = html.Div(f"File loaded successfully: {filename}", className="text-success")

    # 2. Process and Get Results
    fig_processed, num_spots, total_area, total_white_area, ratio_white_to_total = \
        process_image(img_rgb, threshold, threshold_type)

    # 3. Format Stats Output
    stats_num_spots = html.H5(f"Spots Detected: {num_spots}")
    stats_total_area = html.Div(f"Total Image Area: {total_area:,} pixels")
    stats_white_area = html.Div(f"Thresholded White Area: {total_white_area:,} pixels")
    stats_ratio = html.Div(f"Ratio (White/Total): {ratio_white_to_total*100:.4f}%")
    
    return fig_processed, stats_num_spots, stats_total_area, stats_white_area, stats_ratio, status_msg

# Standard practice to run the app locally if executed directly
if __name__ == '__main__':
    app.run(debug=True)
