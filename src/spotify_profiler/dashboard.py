from dash import Dash, html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

class SpotifyDashboard:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.app = Dash(__name__)
        self.setup_layout()
        self.register_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Spotify Listening Analysis Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # File Selection Section
            html.Div([
                html.Div([
                    html.H3("Analysis Files", style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='analysis-file-dropdown',
                        options=[],
                        placeholder="Select an analysis file",
                        style={'marginBottom': '10px'}
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Comparison Files", style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='comparison-file-dropdown',
                        options=[],
                        placeholder="Select a comparison file",
                        style={'marginBottom': '10px'}
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ], style={'marginBottom': '20px'}),

            # Main Content Tabs
            dcc.Tabs([
                # Individual Analysis Tab
                dcc.Tab(label='Individual Analysis', children=[
                    html.Div([
                        html.Div([
                            html.H3("General Metrics", style={'textAlign': 'center'}),
                            dcc.Graph(id='user-metrics-gauge')
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                            html.Div([
                                html.H3("Genre Distribution", style={'textAlign': 'center'}),
                                dcc.Graph(id='genre-pie-chart')
                            ], style={'width': '48%'}),
                            
                            html.Div([
                                html.H3("Top Artists", style={'textAlign': 'center'}),
                                dcc.Graph(id='top-artists-bar')
                            ], style={'width': '48%'})
                        ]),
                        
                        html.Div([
                            html.H3("Listening Activity Patterns", style={'textAlign': 'center'}),
                            dcc.Graph(id='temporal-heatmap')
                        ], style={'marginTop': '20px'})
                    ])
                ]),
                
                # Comparison Tab
                dcc.Tab(label='User Comparison', children=[
                    html.Div([
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                            html.Div([
                                html.H3("Overall Similarity", style={'textAlign': 'center'}),
                                dcc.Graph(id='similarity-gauge')
                            ], style={'width': '48%'}),
                            
                            html.Div([
                                html.H3("Genre Comparison", style={'textAlign': 'center'}),
                                dcc.Graph(id='genre-comparison-radar')
                            ], style={'width': '48%'})
                        ]),
                        
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
                            html.Div([
                                html.H3("Artist Overlap", style={'textAlign': 'center'}),
                                dcc.Graph(id='artist-venn')
                            ], style={'width': '48%'}),
                            
                            html.Div([
                                html.H3("Temporal Pattern Comparison", style={'textAlign': 'center'}),
                                dcc.Graph(id='temporal-comparison')
                            ], style={'width': '48%'})
                        ], style={'marginTop': '20px'})
                    ])
                ])
            ])
        ])

    def register_callbacks(self):
        @self.app.callback(
            [Output('analysis-file-dropdown', 'options'),
             Output('comparison-file-dropdown', 'options')],
            [Input('analysis-file-dropdown', 'value')]  # Dummy input for initial load
        )
        def update_dropdowns(_):
            try:
                analysis_dir = Path('analysis_results')
                comparison_dir = Path('comparison_results')
                
                analysis_files = sorted(analysis_dir.glob('analysis_results_*.json'),
                                     key=lambda x: x.stat().st_mtime, reverse=True)
                comparison_files = sorted(comparison_dir.glob('comparison_*.json'),
                                       key=lambda x: x.stat().st_mtime, reverse=True)
                
                analysis_options = [{'label': f.name, 'value': str(f)} for f in analysis_files]
                comparison_options = [{'label': f.name, 'value': str(f)} for f in comparison_files]
                
                return analysis_options, comparison_options
            except Exception as e:
                self.logger.error(f"Error updating dropdowns: {e}")
                return [], []

        @self.app.callback(
            [Output('genre-pie-chart', 'figure'),
             Output('top-artists-bar', 'figure'),
             Output('temporal-heatmap', 'figure'),
             Output('user-metrics-gauge', 'figure')],
            [Input('analysis-file-dropdown', 'value')]
        )
        def update_analysis_graphs(selected_file):
            if not selected_file:
                raise PreventUpdate
                
            try:
                with open(selected_file, 'r') as f:
                    data = json.load(f)
                    
                return (
                    self._create_genre_pie(data),
                    self._create_artists_bar(data),
                    self._create_temporal_heatmap(data),
                    self._create_metrics_gauge(data)
                )
            except Exception as e:
                self.logger.error(f"Error updating analysis graphs: {e}")
                return [{} for _ in range(4)]

        @self.app.callback(
            [Output('similarity-gauge', 'figure'),
             Output('genre-comparison-radar', 'figure'),
             Output('artist-venn', 'figure'),
             Output('temporal-comparison', 'figure')],
            [Input('comparison-file-dropdown', 'value')]
        )
        def update_comparison_graphs(selected_file):
            if not selected_file:
                raise PreventUpdate
                
            try:
                with open(selected_file, 'r') as f:
                    data = json.load(f)
                    
                return (
                    self._create_similarity_gauge(data),
                    self._create_genre_radar(data),
                    self._create_artist_venn(data),
                    self._create_temporal_comparison(data)
                )
            except Exception as e:
                self.logger.error(f"Error updating comparison graphs: {e}")
                return [{} for _ in range(4)]

    def _create_genre_pie(self, data: Dict) -> go.Figure:
        genres = data['genres']['top_genres']
        values = list(genres.values())
        labels = list(genres.keys())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Genre Distribution",
            showlegend=False,
            height=400
        )
        return fig

    def _create_artists_bar(self, data: Dict) -> go.Figure:
        artists = data['artists']['top_artists']
        
        fig = go.Figure(data=[go.Bar(
            x=list(artists.keys()),
            y=list(artists.values()),
            marker_color='rgb(55, 83, 109)'
        )])
        
        fig.update_layout(
            title="Top Artists by Play Count",
            xaxis_title="Artist",
            yaxis_title="Play Count",
            height=400,
            xaxis={'tickangle': -45}
        )
        return fig

    def _create_temporal_heatmap(self, data: Dict) -> go.Figure:
        temporal = data['temporal_patterns']['distributions']['hourly']
        hours = list(range(24))
        values = [temporal.get(str(hour), 0) for hour in hours]
        
        fig = go.Figure(data=go.Heatmap(
            z=[values],
            x=hours,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title="Listening Activity by Hour",
            xaxis_title="Hour of Day",
            yaxis_showticklabels=False,
            height=300
        )
        return fig

    def _create_metrics_gauge(self, data: Dict) -> go.Figure:
        metrics = data['user_metrics']['listening_stats']
        
        fig = go.Figure()
        
        # Add listening time gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['total_time_hours'],
            domain={'x': [0, 0.5], 'y': [0, 1]},
            title={'text': "Listening Time (Hours)"},
            gauge={
                'axis': {'range': [0, max(100, metrics['total_time_hours'])]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ]
            }
        ))
        
        # Add unique tracks gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['unique_tracks'],
            domain={'x': [0.5, 1], 'y': [0, 1]},
            title={'text': "Unique Tracks"},
            gauge={
                'axis': {'range': [0, max(100, metrics['unique_tracks'])]},
                'bar': {'color': "darkblue"}
            }
        ))
        
        fig.update_layout(height=300)
        return fig

    def _create_similarity_gauge(self, data: Dict) -> go.Figure:
        similarity = data['overall_similarity']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity,
            title={'text': "Overall Similarity Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig

    def _create_genre_radar(self, data: Dict) -> go.Figure:
        genre_sim = data['genre_similarity']
        shared_genres = genre_sim['top_shared_genres'][:5]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[genre[1] for genre in shared_genres],
            theta=[genre[0] for genre in shared_genres],
            fill='toself',
            name='User 1'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[genre[2] for genre in shared_genres],
            theta=[genre[0] for genre in shared_genres],
            fill='toself',
            name='User 2'
        ))
        
        fig.update_layout(
            polar={'radialaxis': {'visible': True, 'range': [0, 100]}},
            showlegend=True,
            height=400
        )
        
        return fig

    def _create_artist_venn(self, data: Dict) -> go.Figure:
        artist_sim = data['artist_similarity']
        
        fig = go.Figure()
        
        # Create a visualization of overlapping artists using circles
        fig.add_shape(
            type="circle",
            x0=-1, y0=-1, x1=1, y1=1,
            fillcolor="blue",
            opacity=0.3,
            line_color="blue",
        )
        
        fig.add_shape(
            type="circle",
            x0=0, y0=-1, x1=2, y1=1,
            fillcolor="red",
            opacity=0.3,
            line_color="red",
        )
        
        # Add annotations
        fig.add_annotation(
            x=-0.5, y=0,
            text=f"User 1\n{len(artist_sim['unique_preferences']['user1_unique'])} artists",
            showarrow=False
        )
        
        fig.add_annotation(
            x=1.5, y=0,
            text=f"User 2\n{len(artist_sim['unique_preferences']['user2_unique'])} artists",
            showarrow=False
        )
        
        fig.add_annotation(
            x=0.5, y=0,
            text=f"Common\n{artist_sim['common_artists_count']} artists",
            showarrow=False
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            height=400
        )
        
        return fig

    def _create_temporal_comparison(self, data: Dict) -> go.Figure:
        temporal = data['temporal_similarity']
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Daily Similarity', 'Hourly Similarity'],
                y=[temporal['daily_similarity'] * 100, temporal['hourly_similarity'] * 100],
                marker_color=['rgb(55, 83, 109)', 'rgb(26, 118, 255)']
            )
        ])
        
        fig.update_layout(
            title="Temporal Pattern Similarity",
            yaxis_title="Similarity (%)",
            yaxis_range=[0, 100],
            height=400
        )
        
        return fig

    def run_server(self, debug=False, port=8050):
        try:
            self.app.run_server(debug=debug, port=port)
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")
            raise
