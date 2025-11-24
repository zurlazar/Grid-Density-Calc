import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import base64
import io
from PIL import Image
import cv2
import math

# --- 1. Global Initialization for Dash and Gunicorn ---
# These must be defined globally for Gunicorn to find the WSGI application object.
# Use a modern, responsive theme like CERULEAN
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server # This is the WSGI object Gunicorn needs to serve the app

# --- 2. Data and Helper Functions ---

def parse_contents(contents, filename):
    """
    Decodes the base64 content from the Dash Upload component into a NumPy array
    (OpenCV BGR format).
    """
    if contents is None:
        return None
    
    try:
        # Get content type and base64 string
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Open image using PIL
        image_stream = io.BytesIO(decoded)
        pil_img = Image.open(image_stream).convert('RGB')
        
        # Convert PIL image to NumPy array (BGR format for OpenCV)
        img_rgb = np.array(pil_img)
        return img_rgb
        
    except Exception as e:
        print(f"Error parsing image contents: {e}")
        return None

def process_image_roi(img_rgb, roi_data, threshold):
    """
    Processes the image within the specified ROI, finds spots, and calculates density statistics.
    
    Args:
        img_rgb (np.array): The full original image array (RGB).
        roi_data (dict or None): Dictionary containing the ROI ranges, or None for full image.
        threshold (int): The binary threshold value (0-255).
        
    Returns:
        tuple: (processed_figure, num_spots, total_roi_area, total_white_area, ratio_white_to_total)
    """
    if img_rgb is None:
        return go.Figure(), 0, 0, 0, 0.0

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    H, W = img_rgb.shape[:2]
    
    # --- 2.1 Determine Coordinates ---
    if roi_data and 'range' in roi_data and 'xaxis.range' in roi_data['range'] and 'yaxis.range' in roi_data['range']:
        x_range = roi_data['range']['xaxis.range']
        y_range = roi_data['range']['yaxis.range']
        
        # Coordinates must be integers and within bounds
        # Note: Plotly's image layout inverts Y axis, so y0 is max and y1 is min visually.
        x0 = max(0, int(math.floor(min(x_range))))
        x1 = min(W, int(math.ceil(max(x_range))))
        y0 = max(0, int(math.floor(min(y_range))))
        y1 = min(H, int(math.ceil(max(y_range))))
        
    else:
        # Full Image
        x0, x1 = 0, W
        y0, y1 = 0, H

    # Ensure valid range
    if x1 <= x0 or y1 <= y0:
        return go.Figure(), 0, 0, 0, 0.0
    
    # --- 2.2 Slice and Process ROI ---
    roi_gray = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    roi_bgr = img_bgr[y0:y1, x0:x1].copy()
    roi_height, roi_width = roi_gray.shape[:2]
    
    total_roi_area = roi_height * roi_width
    
    # Binarize ROI
    _, binary_roi = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate white pixel area
    total_white_area = np.sum(binary_roi == 255)
    
    # Calculate ratio
    ratio_white_to_total = total_white_area / total_roi_area if total_roi_area > 0 else 0.0
    
    # Find Contours in ROI for spot detection
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_spots = len(contours)
    
    # Draw contours on the BGR ROI for display
    cv2.drawContours(roi_bgr, contours, -1, (0, 255, 0), 2)
    
    # Convert back to RGB for Plotly figure
    roi_rgb_marked = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    
    # --- 2.3 Create Processed Figure (Only shows the ROI) ---
    fig_processed = go.Figure(go.Image(z=roi_rgb_marked))
    fig_processed.update_layout(
        title=f"Processed Grid (ROI: {roi_width}x{roi_height})",
        xaxis_title="X (ROI)", yaxis_title="Y (ROI)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_range=[0, roi_width],
        yaxis_range=[roi_height, 0], # Invert Y-axis for image convention
    )
    fig_processed.update_xaxes(showgrid=False, zeroline=False, showticklabels=True)
    fig_processed.update_yaxes(showgrid=False, zeroline=False, showticklabels=True)

    return fig_processed, num_spots, total_roi_area, total_white_area, ratio_white_to_total


# --- 3. Dash Layout ---

app.layout = dbc.Container([
    # Hidden component to store the image data (base64 string)
    dcc.Store(id='uploaded-image-data'),

    dbc.Row([
        dbc.Col(html.H1("Grid Density and Spot Calculator", className="text-center text-primary my-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ), width=12)
    ]),

    # --- Threshold Control Row ---
    dbc.Row([
        dbc.Col(html.Div(id='output-image-upload'), width=12, className="text-center mb-3"),
        dbc.Col(html.H5("Threshold Value (for Binarization):", className="mt-3"), width=12),
        dbc.Col(dcc.Slider(
            id='threshold-slider',
            min=0, max=255, step=1, value=127,
            marks={i: str(i) for i in range(0, 256, 32)},
        ), width=10),
        dbc.Col(html.Div(id='threshold-output', className="font-weight-bold pt-2"), width=2)
    ], className="mb-4 align-items-center"),

    # --- Main Display Row (Adjusted for visual focus) ---
    dbc.Row([
        # LEFT: Original Image with ROI selector (Smaller: width=6)
        dbc.Col(dcc.Graph(id='main-image-graph', config={'scrollZoom': True}), width=6),
        
        # RIGHT: Analysis/Stats/Processed ROI (Larger focus: width=6)
        dbc.Col([
            # Stats Card
            dbc.Card(
                dbc.CardBody([
                    html.H4("Analysis Results (Live ROI)", className="card-title text-success"),
                    html.Div(id='roi-info-text', className="mb-2 text-muted"),
                    html.Div(id='stats-num-spots', className="h5"),
                    html.Div(id='stats-total-area'),
                    html.Div(id='stats-white-area'),
                    html.Div(id='stats-ratio'),
                ]),
                className="mb-4 shadow-sm"
            ),
            
            # Processed ROI Figure Card (Increased height for larger view)
            dbc.Card(
                dbc.CardBody([
                    html.H4("Processed ROI View", className="card-title"),
                    dcc.Graph(id='processed-roi-graph', style={'height': '550px'}) 
                ], className="p-2") # Reduced padding inside card body for graph space
            )
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.P("Instructions: 1. Upload an image. 2. Adjust the threshold. 3. Click and drag on the full image to select a Region of Interest (ROI). 4. Statistics update live for the selected ROI.", className="text-center text-muted"), width=12)
    ])

], fluid=True)


# --- 4. Callbacks ---

# Callback 1: Upload and Store Image Data
@app.callback(
    Output('uploaded-image-data', 'children'),
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def store_data(contents, filename):
    if contents is not None:
        return contents, html.Div(f'File uploaded: {filename}', className="text-success")
    return dash.no_update, html.Div("Awaiting image upload...", className="text-warning")

# Callback 2: Update Threshold Display
@app.callback(
    Output('threshold-output', 'children'),
    Input('threshold-slider', 'value')
)
def update_threshold_output(value):
    return f"{value}"

# Callback 3: Main Analysis and Figure Updates (The Core Logic)
@app.callback(
    Output('main-image-graph', 'figure'),
    Output('processed-roi-graph', 'figure'),
    Output('stats-num-spots', 'children'),
    Output('stats-total-area', 'children'),
    Output('stats-white-area', 'children'),
    Output('stats-ratio', 'children'),
    Output('roi-info-text', 'children'),
    
    Input('uploaded-image-data', 'children'),
    Input('threshold-slider', 'value'),
    Input('main-image-graph', 'relayoutData') # Input for ROI data
)
def update_analysis(img_base64, threshold, relayoutData):
    
    if img_base64 is None:
        # Initial state or no image loaded
        default_fig = go.Figure().update_layout(title="Image Display Area")
        return (default_fig, default_fig, "Spots Detected: 0", 
                "ROI Area: 0 pixels", "White Area: 0 pixels", 
                "Ratio: 0.0000%", "Please load an image to begin.")

    img_rgb = parse_contents(img_base64, "temp")
    
    if img_rgb is None:
        # Error during parsing
        error_fig = go.Figure().update_layout(title="Error: Could not load image")
        return (error_fig, error_fig, "Spots Detected: 0", 
                "ROI Area: 0 pixels", "White Area: 0 pixels", 
                "Ratio: 0.0000%", "Could not process image data.")

    H, W = img_rgb.shape[:2]

    # --- 1. Determine ROI (Robust Key Handling) ---
    roi_data = None
    roi_info_text = "ROI: Full Image (Select by clicking and dragging)"
    x_range, y_range = None, None
    
    # Check for a specific BOX SELECTION or ZOOM/PAN event
    if relayoutData:
        # Check for the range keys which are reliably present during box selection/zoom events
        if 'xaxis.range' in relayoutData and 'yaxis.range' in relayoutData:
            x_range = relayoutData['xaxis.range']
            y_range = relayoutData['yaxis.range']

        # NOTE: Keys like 'xaxis.range[0]' are typically only present during pan/zoom 
        # but not always with the full range definition, leading to KeyError.
        # Focusing on 'xaxis.range' is more robust for range extraction.

    # If x_range and y_range were successfully extracted, set the ROI data
    if x_range and y_range:
        # Pass the extracted ranges to the processing function
        roi_data = {'range': {'xaxis.range': x_range, 'yaxis.range': y_range}}
        
        # Display info: Note that in Plotly's image layout, y-axis is inverted
        x_min_display = int(min(x_range))
        x_max_display = int(max(x_range))
        y_min_display = int(min(y_range))
        y_max_display = int(max(y_range))
        
        # Display Y range from top (max value) to bottom (min value) for visual consistency
        roi_info_text = f"ROI: X[{x_min_display:,}:{x_max_display:,}], Y[{y_max_display:,}:{y_min_display:,}]"


    # --- 2. Process and Get Results ---
    # Pass the image and the potentially defined roi_data to the processor
    fig_processed, num_spots, total_roi_area, total_white_area, ratio_white_to_total = \
        process_image_roi(img_rgb, roi_data, threshold)

    # --- 3. Update Main Image Figure (Include ROI Marker) ---
    fig_main = go.Figure(go.Image(z=img_rgb))
    fig_main.update_layout(
        title="Full Image (Select ROI by clicking and dragging)",
        xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
        dragmode='select', # Enables box selection
        modebar_activecolor='red',
        xaxis_range=[0, W],
        yaxis_range=[H, 0], # Invert Y-axis
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig_main.update_xaxes(showgrid=False, zeroline=False, showticklabels=True)
    fig_main.update_yaxes(showgrid=False, zeroline=False, showticklabels=True)
    
    # Draw the selected ROI box on the main figure for visual confirmation
    if x_range and y_range:
        fig_main.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x_range[0], y0=y_range[1], x1=x_range[1], y1=y_range[0], # y-coords are swapped due to inversion
            line=dict(color="Red", width=3),
            opacity=0.5
        )


    # --- 4. Format Stats Output ---
    stats_num_spots = html.H5(f"Spots Detected: {num_spots}")
    stats_total_area = html.Div(f"ROI Area: {total_roi_area:,} pixels")
    stats_white_area = html.Div(f"White Area: {total_white_area:,} pixels")
    stats_ratio = html.Div([
        "Ratio (White/Total): ", 
        html.B(f"{ratio_white_to_total*100:.4f}%", className="text-danger")
    ])
    
    return fig_main, fig_processed, stats_num_spots, stats_total_area, stats_white_area, stats_ratio, roi_info_text

# Standard practice to run the app locally if executed directly
if __name__ == '__main__':
    app.run_server(debug=True)
