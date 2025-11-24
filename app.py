import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import os

# --- Configuration & Initialization ---
# Use a bootsrap theme for a modern look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server # This is needed for Render deployment

IMAGE_DIR = 'images'
# Define your sample images here. Keys are filenames, values are display names.
SAMPLE_IMAGES = {
    'sample_1.png': 'Default Image 1 (Bright Spots)',
    'sample_2.jpg': 'Default Image 2 (Test Image)',
}

# --- Utility Functions ---

def load_image_as_numpy(image_path):
    """Loads a local image file into an RGB numpy array."""
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def parse_contents(contents, filename):
    """Decodes the uploaded base64 image content into an RGB numpy array."""
    if contents is None:
        return None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded))
        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def process_image_roi(img_rgb, roi_data, threshold):
    """
    Processes the image within the specified ROI (or the whole image) 
    and calculates metrics and returns the marked figure.
    """
    if img_rgb is None:
        return go.Figure(), 0, 0, 0, 0

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray_img.shape
    
    # 1. Determine ROI boundaries
    x0, y0, x1, y1 = 0, 0, W, H # Default to full image
    
    if roi_data and 'range' in roi_data and 'xaxis.range[0]' in roi_data['range']:
        # Plotly coordinates (x, y) must be clamped to image dimensions
        x_range = roi_data['range']['xaxis.range']
        y_range = roi_data['range']['yaxis.range']
        x0, x1 = int(max(0, min(x_range))), int(min(W, max(x_range)))
        y0, y1 = int(max(0, min(y_range))), int(min(H, max(y_range)))
        
        # Ensure correct order: x0 < x1, y0 < y1
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

    # 2. Extract ROI and calculate stats
    roi_gray = gray_img[y0:y1, x0:x1]
    roi_bgr = img_bgr[y0:y1, x0:x1].copy()
    
    total_roi_area = roi_gray.size
    
    if total_roi_area == 0:
        return go.Figure(), 0, 0, 0, 0

    # Binarize ROI
    _, binary_roi = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find Contours in ROI
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_spots = len(contours)
    total_white_area = np.sum(binary_roi == 255)
    ratio_white_to_total = total_white_area / total_roi_area
    
    # 3. Mark the image
    center_x, center_y = [], []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            # Centroid (relative to ROI)
            cX_roi = int(M["m10"] / M["m00"])
            cY_roi = int(M["m01"] / M["m00"])

            center_x.append(cX_roi)
            center_y.append(cY_roi)

            # Draw contour and center on the ROI copy (in BGR)
            cv2.drawContours(roi_bgr, [contour], -1, (0, 255, 0), 1) 
            cv2.circle(roi_bgr, (cX_roi, cY_roi), 3, (0, 255, 0), -1) 

    # 4. Create Plotly Figure (only for the ROI, showing the marked image)
    roi_rgb_marked = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    
    fig = go.Figure()
    fig.add_trace(go.Image(z=roi_rgb_marked))
    
    # Overlay centers
    fig.add_trace(go.Scatter(
        x=center_x, 
        y=center_y, 
        mode='markers', 
        marker=dict(size=8, color='lime', symbol='circle', line=dict(width=1, color='black')),
        name='Center of Mass',
        hoverinfo='text',
        hovertext=[f'Spot: ({x}, {y})' for x, y in zip(center_x, center_y)]
    ))
    
    fig.update_layout(
        title=f"Analysis of Selected ROI (Threshold: {threshold})",
        xaxis_title="X (relative to ROI)",
        yaxis_title="Y (relative to ROI)",
        xaxis_range=[0, roi_rgb_marked.shape[1]],
        yaxis_range=[roi_rgb_marked.shape[0], 0],
        dragmode='select', # Allows the user to select a new ROI on this figure too
        showlegend=False
    )

    return fig, num_spots, total_roi_area, total_white_area, ratio_white_to_total

# --- Dash Layout ---

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Interactive Spot Image Analyzer 🔬", className="text-center my-4"), width=12)
    ]),
    
    # Input/Control Row
    dbc.Row([
        dbc.Col([
            # 1. File Upload
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                },
                multiple=False
            ),
            # Hidden div to store the base64 image data
            html.Div(id='uploaded-image-data', style={'display': 'none'}),
        ], width=4),
        
        dbc.Col([
            # 2. Sample Image Selector
            dcc.Dropdown(
                id='sample-image-dropdown',
                options=[{'label': v, 'value': k} for k, v in SAMPLE_IMAGES.items()],
                placeholder="--- Or Select a Sample Image ---",
            ),
        ], width=4),
        
        dbc.Col([
            # 3. Threshold Slider
            html.Label("Binarization Threshold (0-255):"),
            dcc.Slider(
                id='threshold-slider',
                min=0, max=255, step=1, value=150,
                marks={0: '0', 255: '255', 150: '150'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], width=4),
    ], className="mb-4"),

    dbc.Row([
        # Main Figure (Shows full image, allows ROI selection)
        dbc.Col(dcc.Graph(id='main-image-graph', config={'scrollZoom': True}), width=8),
        
        # Live Stats and Results
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Analysis Results (Live ROI)", className="card-title"),
                    html.P(id='stats-roi-info', style={'font-size': '0.8rem', 'color': '#777'}),
                    html.Hr(),
                    html.H5(id='stats-num-spots'),
                    html.Div(id='stats-total-area'),
                    html.Div(id='stats-white-area'),
                    html.Div(id='stats-ratio', className="h3 text-primary"),
                ]),
                className="mb-4"
            ),
            # Final Processed ROI Figure
            dbc.Card(
                dbc.CardBody([
                    html.H4("Processed ROI View", className="card-title"),
                    dcc.Graph(id='processed-roi-graph', style={'height': '350px'})
                ])
            )
        ], width=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.P("Drag the mouse on the main image to select a Region of Interest (ROI) and see the live update of statistics.", className="text-center font-italic"), width=12),
        # Signature
        dbc.Col(html.P("Made by Zur Lazar, Nov 24th, 2025", className="text-right text-muted mt-2"), width=12),
    ])

], fluid=True)


# --- Callbacks ---

# Callback 1: Load image data (from upload or sample)
@app.callback(
    Output('uploaded-image-data', 'children'),
    [Input('upload-image', 'contents'),
     Input('sample-image-dropdown', 'value')],
    [State('upload-image', 'filename')]
)
def load_image_data(uploaded_contents, sample_filename, uploaded_filename):
    """Loads image data into the hidden div."""
    ctx = dash.callback_context
    if not ctx.triggered:
        # Default to the first sample image on initial load
        if SAMPLE_IMAGES:
            sample_path = os.path.join(IMAGE_DIR, list(SAMPLE_IMAGES.keys())[0])
            img_rgb = load_image_as_numpy(sample_path)
            if img_rgb is not None:
                # Convert numpy array back to base64 string for storage in the hidden div
                img = Image.fromarray(img_rgb)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode()
        return None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'upload-image' and uploaded_contents:
        # Uploaded file takes priority
        return uploaded_contents
    
    elif trigger_id == 'sample-image-dropdown' and sample_filename:
        # Load sample file
        sample_path = os.path.join(IMAGE_DIR, sample_filename)
        img_rgb = load_image_as_numpy(sample_path)
        if img_rgb is not None:
            # Convert numpy array back to base64 string for storage
            img = Image.fromarray(img_rgb)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode()
            
    return None


# Callback 2: Display the main image and handle ROI/Threshold changes
@app.callback(
    [Output('main-image-graph', 'figure'),
     Output('processed-roi-graph', 'figure'),
     Output('stats-num-spots', 'children'),
     Output('stats-total-area', 'children'),
     Output('stats-white-area', 'children'),
     Output('stats-ratio', 'children'),
     Output('stats-roi-info', 'children')],
    [Input('uploaded-image-data', 'children'),
     Input('threshold-slider', 'value'),
     Input('main-image-graph', 'relayoutData')]
)
def update_analysis(img_base64, threshold, relayoutData):
    """Updates the main figure, the processed ROI figure, and all stats."""
    
    if img_base64 is None:
        # Return empty figures and stats if no image is loaded
        return go.Figure(), go.Figure(), "No image loaded.", "", "", "", "Please load an image to begin."

    # Parse the base64 image data
    img_rgb = parse_contents(img_base64, "temp")
    if img_rgb is None:
        return go.Figure(), go.Figure(), "Error loading image.", "", "", "", "Could not process image data."

    # Determine the ROI from relayoutData (the user's drag selection)
    roi_data = None
    roi_info_text = "ROI: Full Image"

    # Check for a specific BOX SELECTION event (which is the most reliable way to get ROI data)
    if relayoutData and 'xaxis.range' in relayoutData and 'yaxis.range' in relayoutData:
        # 1. This means the user performed a box selection (dragmode='select')
        #    The structure is correct: relayoutData={'xaxis.range': [x0, x1], 'yaxis.range': [y0, y1]}
        
        x_range = relayoutData['xaxis.range']
        y_range = relayoutData['yaxis.range']
        
        # Construct roi_data for the processing function
        roi_data = {'range': {'xaxis.range': x_range, 'yaxis.range': y_range}}
        
        # Display info: Note that in Plotly's layout, y-axis is inverted (y1 to y0)
        roi_info_text = f"ROI: X[{int(x_range[0]):,}:{int(x_range[1]):,}], Y[{int(y_range[1]):,}:{int(y_range[0]):,}]"
        
    elif relayoutData and any(key.endswith('.range[0]') for key in relayoutData):
        # 2. This handles general panning or zooming events which change the range.
        #    This is typically NOT a full ROI selection but a viewport change.
        #    We must extract the full range from the keys provided.
        #    If the user has zoomed/panned, use the new viewport as the ROI.
        
        # Find the updated range keys (xaxis.range, yaxis.range)
        x_key = next((k for k in relayoutData if k.startswith('xaxis.range')), None)
        y_key = next((k for k in relayoutData if k.startswith('yaxis.range')), None)
        
        if x_key and y_key:
            x_range = relayoutData[x_key]
            y_range = relayoutData[y_key]

            # Construct roi_data for the processing function
            roi_data = {'range': {'xaxis.range': x_range, 'yaxis.range': y_range}}
            
            roi_info_text = f"ROI: Zoom/Pan X[{int(x_range[0]):,}:{int(x_range[1]):,}], Y[{int(y_range[1]):,}:{int(y_range[0]):,}]"

    # --- Process and Get Results ---
    fig_processed, num_spots, total_roi_area, total_white_area, ratio_white_to_total = \
        process_image_roi(img_rgb, roi_data, threshold)

    # --- Update Main Image Figure (Always displays the full image with ROI marker) ---
    fig_main = go.Figure(go.Image(z=img_rgb))
    fig_main.update_layout(
        title="Full Image (Select ROI by clicking and dragging)",
        xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
        dragmode='select', # This is crucial for ROI selection
        modebar_activecolor='red',
        xaxis_range=[0, img_rgb.shape[1]],
        yaxis_range=[img_rgb.shape[0], 0], # Invert Y-axis
    )
    
    # Optional: Draw the selected ROI box on the main figure
    if roi_data:
        x0, x1 = relayoutData['xaxis.range']
        y0, y1 = relayoutData['yaxis.range']
        fig_main.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x0, y0=y1, x1=x1, y1=y0,
            line=dict(color="Red", width=2),
            opacity=0.5
        )


    # --- Format Stats Output ---
    stats_num_spots = html.H5(f"Spots Detected: {num_spots}")
    stats_total_area = html.Div(f"ROI Area: {total_roi_area:,} pixels")
    stats_white_area = html.Div(f"White Area: {total_white_area:,} pixels")
    stats_ratio = html.Div(f"Ratio (White/Total): {ratio_white_to_total*100:.4f}%")
    
    return fig_main, fig_processed, stats_num_spots, stats_total_area, stats_white_area, stats_ratio, roi_info_text

# Run the app (for local testing only)
if __name__ == '__main__':
    # Create the images directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created directory: {IMAGE_DIR}. Please place your sample images inside.")
    
    print("Starting Dash server. Go to http://127.0.0.1:8050/")
    app.run_server(debug=True)

