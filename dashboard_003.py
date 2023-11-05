import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
from PIL import Image
import dash_bootstrap_components as dbc

# Create the app
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

# Generate data
# Load an Excel file using Pandas
file_path = 'EO7 Work Sample Stimulus Data.xlsx'  # Replace 'your_file.xlsx' with the path to your Excel file

# Use Pandas read_excel to read the Excel file into a DataFrame and specify the header row
df = pd.read_excel(file_path, header=2)  # Set header=2 to use the third row as column names (zero-based index)
#correst 24:00 dataformat conversion error
df['Time'] = df['Time'].replace(datetime.datetime(1900, 1, 1, 0, 0), datetime.time(0, 0))
# Convert 'Time' to datetime format (Assuming Time is in HH:MM:SS format)
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
# Combine 'Date' and 'Time' into a new 'Datetime' column
timestamp_df=pd.DataFrame(df['Date'] + pd.to_timedelta(df['Time'].astype(str)), columns=['Datetime'])
# add 1 day to the timestamp with 0 hour
index=timestamp_df['Datetime'].dt.hour==0
timestamp_df['Datetime'][index]=timestamp_df['Datetime'][index]+pd.Timedelta(days=1)
# Convert to strings for the slider marks
timestamp_str_pandasSeries = (df['Date'] + pd.to_timedelta(df['Time'].astype(str))).dt.strftime(date_format='%Y-%m-%d %H:%M:%S')

# Filter the columns starting with 'WOLLONGONG' and create a new DataFrame
WOLLONGONG_df = df.filter(regex='^WOLLONGONG').interpolate(method='linear')
BRINGELLY_df = df.filter(regex='^BRINGELLY').interpolate(method='linear')
WYONG_df = df.filter(regex='^WYONG').interpolate(method='linear')

#select sensor within dataframe
sensor_string=['HUMID', 'NEPH','OZONE','PM10','PM2.5','TEMP','WDR', 'WSP']
unit_dict={'HUMID':'[%]', 'NEPH':'[bsp]','OZONE':'[pphm]','PM10':'[µg/m³]','PM2.5':'[µg/m³]','TEMP':'[°C]','WDR':'[°]', 'WSP':'[m/s]'}

#coordinate of weather station
location_df = {
    'name': ['Bringelly', 'Wyong', 'Wollongong'],
    'latitude': [-33.921667, -33.282000, -34.427222],
    'longitude': [150.732222, 151.418000, 150.893889]
}

#  three locations coordinates
bringelly_coords = np.array((150.732222, -33.921667))
wyong_coords = np.array((151.418000, -33.282000))
wollongong_coords = np.array((150.893889, -34.427222))
# Combine the coordinates 
coords = np.array([bringelly_coords, wyong_coords, wollongong_coords])

# Define the grid resolution
x_grid = np.linspace(min(coords[:, 0]), max(coords[:, 0]), 100)  # Longitudes
y_grid = np.linspace(min(coords[:, 1]), max(coords[:, 1]), 100)  # Latitudes
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Set up the layout
app.layout = html.Div([
    
    html.Div([
        dcc.Graph(id='scatter-plot',
                  style={'width': '90%', 'height': '90%'}),
    ], style={'width': '40%', 'height': '90%', 'display': 'inline-block', 'vertical-align': 'top','padding-top': '5%','padding-left': '5%'}),
    
    html.Div([
        dcc.Graph(id='color-bar',
                  style={'width': '90%', 'height': '90%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'width': '12%', 'height': '90%', 'display': 'inline-block', 'vertical-align': 'top','padding-top': '5%', 'padding-right': '1%'}),

    html.Div([
        html.Label('Select observations index:'),
        dcc.Slider(
            id='observations-index-slider',
            min=0,
            max=len(timestamp_str_pandasSeries) - 1,
            step=1,
            value=100,
            marks={0: timestamp_str_pandasSeries[0], len(timestamp_str_pandasSeries) - 1: timestamp_str_pandasSeries[len(timestamp_str_pandasSeries) - 1],},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Label('Select Sensor:'),
        dcc.Dropdown(
            id='sensor-dropdown',
            options=[{'label': sensor, 'value': sensor} for sensor in sensor_string],
            value=sensor_string[2]  # Default value (you can change this)
        ),
        dcc.Graph(id='time-serie',
                  style={'width': '100%', 'height': '77%','padding-top': '22%'})
        ],
        style={'width': '43%', 'height': '90%', 'display': 'inline-block', 
              'vertical-align': 'top', 'padding-top': '5%'})
], style={'height': '100vh'})


@app.callback(
    Output('scatter-plot', 'figure'),
    Output('color-bar', 'figure'),
    Output('time-serie', 'figure'),
    Input('sensor-dropdown', 'value'),
    Input('observations-index-slider', 'value')
)
def update_graph(selected_sensor, selected_observations_index):
    BRINGELLY_filtered = df.filter(regex=f'^BRINGELLY.*{selected_sensor}').interpolate(method='linear')
    WYONG_filtered = df.filter(regex=f'^WYONG.*{selected_sensor}').interpolate(method='linear')
    WOLLONGONG_filtered = df.filter(regex=f'^WOLLONGONG.*{selected_sensor}').interpolate(method='linear')
    observations = np.array([
        BRINGELLY_filtered.values,
        WYONG_filtered.values,
        WOLLONGONG_filtered.values
    ])
    
    df_4_line=pd.concat([timestamp_df, BRINGELLY_filtered, WYONG_filtered, WOLLONGONG_filtered], axis=1)
    x = [df_4_line['Datetime'][selected_observations_index]] * 3  # Repeating the selected timestamp three times
    y = observations[:, selected_observations_index, -1].tolist()  
    ts_fig = px.line(df_4_line, x='Datetime', y=[col for col in df_4_line.columns if col != 'Datetime_df'])
    ts_fig.add_scatter(x=x, y=y, mode='markers', name='current', marker=dict(size=15, symbol='x', color='black'))

    
    rbf = Rbf(coords[:, 0], coords[:, 1], observations[:, selected_observations_index], function='linear')
    z_mesh = rbf(x_mesh, y_mesh)

    clip_lower = 0.95 * observations[:, selected_observations_index].min()
    clip_upper = 1.05 * observations[:, selected_observations_index].max()
    z_mesh = np.clip(z_mesh, clip_lower, clip_upper)
    
    # Create a a LUT for RGB 
    column_1 = np.linspace(255,0,256)   # First column filled with zeros
    column_2 = np.zeros((256,))   # Second column filled with ones
    column_3 = np.linspace(0,255,256) # Third column filled with threes
    colorLut = np.vstack([column_1, column_2, column_3])
    colorLut=colorLut.transpose().astype(int)

    # create the rgb image
    rgba=np.zeros((z_mesh.shape[0], z_mesh.shape[1], 4))
    for row in  np.arange(rgba.shape[0]):
        for col in np.arange(rgba.shape[1]) :
            z_pixel = (z_mesh[row,col] - z_mesh.min()) * 255.0 / (z_mesh.max() - z_mesh.min())
            rgba[row,col,0:3]=colorLut[int(z_pixel)]
            rgba[row,col,3]=0 if z_pixel>= 255 else 127 #transparency
    img = Image.fromarray(rgba.astype("uint8"),'RGBA')
    
    # plot figure
    location_df['size']=[10,10,10]
    scatter_mapbox_fig = px.scatter_mapbox(location_df, lat='latitude', lon='longitude',
                                           size='size', zoom=8, color="name", color_discrete_sequence=['blue', 'red', 'green'],
                                           title= 'DATE : '+ timestamp_str_pandasSeries[selected_observations_index])
    scatter_mapbox_fig.update_layout(mapbox_style='open-street-map',
                    mapbox_layers = [
                {
                    'sourcetype': 'image',
                    'source': img,
                    'coordinates': [[x_grid[0], y_grid[0]],[x_grid[-1], y_grid[0]],[x_grid[-1], y_grid[-1]],[x_grid[0], y_grid[-1]]],
                }]
                    )
     
    colorbar_fig = px.imshow(np.tile(colorLut[:,0], (10, 1)).transpose(), color_continuous_scale='bluered_r')
    colorbar_fig.update_xaxes(visible=False)
    colorbar_fig.update_layout()
    colorbar_fig.update_yaxes(title=f'{selected_sensor} {unit_dict[selected_sensor]}')
    colorbar_fig.layout.coloraxis.showscale = False
    colorbar_yLabel= np.linspace(z_mesh.max(),z_mesh.min(), 5).astype(int)
    colorbar_fig.update_yaxes(tickvals=[0, 63,127,191,255], ticktext=colorbar_yLabel)
    

    return scatter_mapbox_fig, colorbar_fig, ts_fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)