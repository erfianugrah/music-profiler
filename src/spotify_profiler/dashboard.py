from dash import Dash, html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

def create_empty_figure(title: str = "No data selected") -> go.Figure:
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

class SpotifyDashboard:
    def __init__(self, initial_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.initial_file = str(initial_file) if initial_file else None
        self.app = Dash(__name__)
        self.empty_figure = create_empty_figure()
        
        # Theme colors - updated for better contrast
        self.colors = {
            'background': '#1A1A1A',  # Darker background
            'card_background': '#2A2A2A',  # Slightly lighter for cards
            'text': '#FFFFFF',
            'muted_text': '#CCCCCC',
            'primary': '#1DB954',  # Spotify green
            'secondary': '#1ED760',  # Lighter Spotify green
            'accent': '#2ECC71',
            'neutral': '#404040',
            'plot_background': '#2A2A2A',
            'grid': '#404040'
        }

        self.plot_template = {
            'layout': {
                'paper_bgcolor': self.colors['card_background'],
                'plot_bgcolor': self.colors['plot_background'],
                'font': {'color': self.colors['text']},
                'xaxis': {'gridcolor': self.colors['grid']},
                'yaxis': {'gridcolor': self.colors['grid']}
            }
        }
        self.setup_layout()
        self.register_callbacks()

    def setup_layout(self):
        analysis_dir = Path('analysis_results')
        comparison_dir = Path('comparison_results')
        
        analysis_files = sorted(analysis_dir.glob('analysis_results_*.json'),
                              key=lambda x: x.stat().st_mtime, reverse=True)
        comparison_files = sorted(comparison_dir.glob('comparison_*.json'),
                                key=lambda x: x.stat().st_mtime, reverse=True)
        
        analysis_options = [{'label': f.name, 'value': str(f)} for f in analysis_files]
        comparison_options = [{'label': f.name, 'value': str(f)} for f in comparison_files]

        self.app.layout = html.Div([
            html.H1("Spotify Listening Analysis Dashboard", 
                   style={
                       'textAlign': 'center',
                       'padding': '20px',
                       'backgroundColor': self.colors['card_background'],
                       'margin': '0 0 20px 0',
                       'color': self.colors['text'],
                       'fontFamily': 'Arial, sans-serif'
                   }),
            
            # File Selection Section
            html.Div([
                html.Div([
                    html.H3("Analysis Files",
                           style={'color': self.colors['text']}),
                    dcc.Dropdown(
                        id='analysis-file-dropdown',
                        options=analysis_options,
                        placeholder="Select an analysis file",
                        className='dropdown-dark'
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Comparison Files",
                           style={'color': self.colors['text']}),
                    dcc.Dropdown(
                        id='comparison-file-dropdown',
                        options=comparison_options,
                        placeholder="Select a comparison file",
                        className='dropdown-dark'
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ], style={
                'padding': '20px',
                'backgroundColor': self.colors['card_background'],
                'marginBottom': '20px',
                'borderRadius': '5px'
            }),

            # Tabs
            dcc.Tabs([
                dcc.Tab(
                    label='Individual Analysis',
                    children=self._create_analysis_layout(),
                    style=self._tab_style(),
                    selected_style=self._tab_selected_style()
                ),
                dcc.Tab(
                    label='User Comparison',
                    children=self._create_comparison_layout(),
                    style=self._tab_style(),
                    selected_style=self._tab_selected_style()
                ),
            ], style={
                'marginTop': '20px',
                'backgroundColor': self.colors['background']
            })
        ], style={
            'backgroundColor': self.colors['background'],
            'minHeight': '100vh',
            'padding': '20px'
        })

    def _standard_layout_settings(self):
        return {
            'paper_bgcolor': self.colors['card_background'],
            'plot_bgcolor': self.colors['plot_background'],
            'font': {'color': self.colors['text'], 'family': 'Arial, sans-serif'},
            'height': 400,
            'margin': dict(l=50, r=50, t=50, b=50),
            'xaxis': {
                'gridcolor': self.colors['neutral'],
                'zerolinecolor': self.colors['neutral']
            },
            'yaxis': {
                'gridcolor': self.colors['neutral'],
                'zerolinecolor': self.colors['neutral']
            }
        }

    def _create_analysis_layout(self):
        return html.Div([
            html.Div([
                html.H3("General Metrics", style=self._section_title_style()),
                dcc.Graph(id='user-metrics-gauge')
            ], style=self._card_style()),
            
            html.Div([
                html.Div([
                    html.H3("Genre Distribution", style=self._section_title_style()),
                    dcc.Graph(id='genre-pie-chart')
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H3("Top Artists", style=self._section_title_style()),
                    dcc.Graph(id='top-artists-bar')
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),
            
            html.Div([
                html.H3("Listening Activity Patterns", style=self._section_title_style()),
                dcc.Graph(id='temporal-heatmap')
            ], style=self._card_style())
        ])

    def _create_comparison_layout(self):
        return html.Div([
            html.Div([
                html.Div([
                    html.H3("Overall Similarity", style=self._section_title_style()),
                    dcc.Graph(id='similarity-gauge')
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H3("Genre Comparison", style=self._section_title_style()),
                    dcc.Graph(id='genre-comparison-radar')
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            
            html.Div([
                html.Div([
                    html.H3("Artist Overlap", style=self._section_title_style()),
                    dcc.Graph(id='artist-venn')
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H3("Temporal Comparison", style=self._section_title_style()),
                    dcc.Graph(id='temporal-comparison')
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'})
        ])

    def _card_style(self):
        return {
            'backgroundColor': self.colors['background'],
            'padding': '20px',
            'marginTop': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }

    def _section_title_style(self):
        return {
            'color': self.colors['text'],
            'marginBottom': '15px',
            'fontSize': '1.2em',
            'fontFamily': 'Arial, sans-serif'
        }

    def _tab_style(self):
        return {
            'padding': '12px',
            'backgroundColor': self.colors['background'],
            'color': self.colors['text'],
            'border': f'1px solid {self.colors["neutral"]}',
            'borderRadius': '5px 5px 0 0',
            'marginRight': '2px'
        }

    def _tab_selected_style(self):
        return {
            'padding': '12px',
            'backgroundColor': self.colors['background'],
            'color': self.colors['primary'],
            'border': f'1px solid {self.colors["primary"]}',
            'borderBottom': 'none',
            'borderRadius': '5px 5px 0 0',
            'marginRight': '2px'
        }

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
                return [create_empty_figure("Select an analysis file to view data") for _ in range(4)]
                
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
                return [create_empty_figure(f"Error: {str(e)}") for _ in range(4)]

        @self.app.callback(
            [Output('similarity-gauge', 'figure'),
             Output('genre-comparison-radar', 'figure'),
             Output('artist-venn', 'figure'),
             Output('temporal-comparison', 'figure')],
            [Input('comparison-file-dropdown', 'value')]
        )
        def update_comparison_graphs(selected_file):
            if not selected_file:
                return [create_empty_figure("Select a comparison file to view data") for _ in range(4)]
                
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
                return [create_empty_figure(f"Error: {str(e)}") for _ in range(4)]

    def _create_genre_pie(self, data: Dict) -> go.Figure:
        genres = data['genres']['top_genres']
        values = list(genres.values())
        labels = list(genres.keys())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='inside',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            )
        )])
        
        fig.update_layout(
            title="Genre Distribution",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig

    def _create_artists_bar(self, data: Dict) -> go.Figure:
        artists = data['artists']['top_artists']
        
        fig = go.Figure(data=[go.Bar(
            x=list(artists.keys()),
            y=list(artists.values()),
            marker_color=self.colors['primary'],
            text=list(artists.values()),
            textposition='auto',
        )])
        
        fig.update_layout(
            title="Top Artists by Play Count",
            xaxis_title="Artist",
            yaxis_title="Play Count",
            height=400,
            margin=dict(l=50, r=50, t=50, b=100),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'tickangle': -45},
            showlegend=False
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
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={'showticklabels': False}
        )
        return fig

    def _create_metrics_gauge(self, data: Dict) -> go.Figure:
        metrics = data['user_metrics']['listening_stats']
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['total_time_hours'],
            domain={'x': [0, 0.5], 'y': [0, 1]},
            title={'text': "Listening Time (Hours)"},
            gauge={
                'axis': {'range': [0, max(100, metrics['total_time_hours'])]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 100], 'color': 'gray'}
                ]
            }
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['unique_tracks'],
            domain={'x': [0.5, 1], 'y': [0, 1]},
            title={'text': "Unique Tracks"},
            gauge={
                'axis': {'range': [0, max(100, metrics['unique_tracks'])]},
                'bar': {'color': self.colors['primary']}
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Arial, sans-serif')
        )
        return fig

    def _create_similarity_gauge(self, data: Dict) -> go.Figure:
        similarity = data['overall_similarity']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity,
            title={'text': "Overall Similarity Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [0, 33], 'color': 'lightgray'},
                    {'range': [33, 66], 'color': 'gray'},
                    {'range': [66, 100], 'color': 'darkgray'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Arial, sans-serif')
        )
        return fig

    def _create_genre_radar(self, data: Dict) -> go.Figure:
        genre_sim = data['genre_similarity']
        shared_genres = genre_sim['top_shared_genres'][:5]  # Top 5 genres
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[genre[1] for genre in shared_genres],
            theta=[genre[0] for genre in shared_genres],
            fill='toself',
            name='User 1',
            line_color=self.colors['primary']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[genre[2] for genre in shared_genres],
            theta=[genre[0] for genre in shared_genres],
            fill='toself',
            name='User 2',
            line_color=self.colors['secondary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Arial, sans-serif')
        )
        return fig

    def _create_artist_venn(self, data: Dict) -> go.Figure:
        artist_sim = data['artist_similarity']
        user1_unique = len(artist_sim['unique_preferences']['user1_unique'])
        user2_unique = len(artist_sim['unique_preferences']['user2_unique'])
        common_count = artist_sim['common_artists_count']
        
        fig = go.Figure()
        
        # Create circles for Venn diagram
        fig.add_shape(
            type="circle",
            x0=-1, y0=-1, x1=1, y1=1,
            fillcolor=self.colors['primary'],
            opacity=0.3,
            line_color=self.colors['primary'],
            layer='below'
        )
        
        fig.add_shape(
            type="circle",
            x0=0, y0=-1, x1=2, y1=1,
            fillcolor=self.colors['secondary'],
            opacity=0.3,
            line_color=self.colors['secondary'],
            layer='below'
        )
        
        # Add annotations
        fig.add_annotation(
            x=-0.5, y=0,
            text=f"User 1<br>{user1_unique} artists",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.add_annotation(
            x=1.5, y=0,
            text=f"User 2<br>{user2_unique} artists",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.add_annotation(
            x=0.5, y=0,
            text=f"Common<br>{common_count} artists",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-1.5, 2.5]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[-1.5, 1.5]
            ),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Arial, sans-serif')
        )
        return fig

    def _create_temporal_comparison(self, data: Dict) -> go.Figure:
        temporal = data['temporal_similarity']
        
        fig = go.Figure(data=[go.Bar(
            x=['Daily Similarity', 'Hourly Similarity'],
            y=[temporal['daily_similarity'] * 100, temporal['hourly_similarity'] * 100],
            marker_color=[self.colors['primary'], self.colors['secondary']],
            text=[f"{temporal['daily_similarity']*100:.1f}%", 
                  f"{temporal['hourly_similarity']*100:.1f}%"],
            textposition='auto',
        )])
        
        fig.update_layout(
            title="Temporal Pattern Similarity",
            yaxis_title="Similarity (%)",
            yaxis_range=[0, 100],
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Arial, sans-serif'),
            showlegend=False
        )
        return fig

    def run_server(self, debug=False, port=8050, host='localhost', use_reloader=False):
        try:
            self.app.run_server(
                debug=debug,
                port=port,
                host=host,
                use_reloader=use_reloader  # This is important
            )
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dashboard = SpotifyDashboard()
    dashboard.run_server(debug=True)
