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
        font=dict(size=16, color="#F8F8F2")
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
        
        self.colors = {
            'background': '#1E1E2E',      # Darker background
            'card_background': '#2B2B40',  # Card background
            'text': '#F8F8F2',            # Light text
            'muted_text': '#BDC0CE',      # Muted text
            'primary': '#8BE9FD',         # Cyan
            'secondary': '#50FA7B',       # Green
            'accent': '#BD93F9',          # Purple
            'highlight': '#FF79C6',       # Pink
            'plot_background': '#1E1E2E',
            'grid': '#44475A',
            'dropdown_bg': '#2B2B40',
            'dropdown_text': '#F8F8F2'
        }

        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .Select-control, .Select-menu-outer {
                        background-color: #2B2B40 !important;
                        border: 1px solid #6272A4 !important;
                    }
                    .Select-value-label, .Select input, .Select-menu {
                        color: #F8F8F2 !important;
                    }
                    .Select-menu-outer {
                        background-color: #2B2B40 !important;
                    }
                    .Select-option {
                        background-color: #2B2B40 !important;
                        color: #F8F8F2 !important;
                    }
                    .Select-option:hover {
                        background-color: #44475A !important;
                    }
                    .Select-placeholder, .Select--single > .Select-control .Select-value {
                        color: #F8F8F2 !important;
                    }
                    .description-box {
                        background-color: #2B2B40;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                        color: #F8F8F2;
                        border: 1px solid #6272A4;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
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
            html.H1("Music Listening Analysis Dashboard", 
                   style={
                       'textAlign': 'center',
                       'padding': '20px',
                       'backgroundColor': self.colors['card_background'],
                       'margin': '0 0 20px 0',
                       'color': self.colors['text'],
                       'fontFamily': 'monospace',
                       'borderRadius': '8px',
                   }),
            
            html.Div([
                html.Div([
                    html.H3("Analysis Files",
                           style={'color': self.colors['text']}),
                    html.Div(
                        "Select a file to view detailed analysis of individual listening patterns.",
                        className='description-box'
                    ),
                    dcc.Dropdown(
                        id='analysis-file-dropdown',
                        options=analysis_options,
                        placeholder="Select an analysis file",
                        style={'color': self.colors['dropdown_text']}
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Comparison Files",
                           style={'color': self.colors['text']}),
                    html.Div(
                        "Select a file to compare listening patterns between two users.",
                        className='description-box'
                    ),
                    dcc.Dropdown(
                        id='comparison-file-dropdown',
                        options=comparison_options,
                        placeholder="Select a comparison file",
                        style={'color': self.colors['dropdown_text']}
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ], style={
                'padding': '20px',
                'backgroundColor': self.colors['card_background'],
                'marginBottom': '20px',
                'borderRadius': '8px',
            }),

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
            'padding': '20px',
            'fontFamily': 'monospace'
        })

    def _create_analysis_layout(self):
        return html.Div([
            html.Div([
                html.H3("General Metrics", style=self._section_title_style()),
                html.Div(
                    "Overview of total listening time and unique tracks in your library.",
                    className='description-box'
                ),
                dcc.Graph(id='user-metrics-gauge')
            ], style=self._card_style()),
            
            html.Div([
                html.Div([
                    html.H3("Genre Distribution", style=self._section_title_style()),
                    html.Div(
                        "Breakdown of your music genres by percentage of total listening time.",
                        className='description-box'
                    ),
                    dcc.Graph(id='genre-pie-chart')
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H3("Top Artists", style=self._section_title_style()),
                    html.Div(
                        "Your most played artists ranked by play count.",
                        className='description-box'
                    ),
                    dcc.Graph(id='top-artists-bar')
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),
            
            html.Div([
                html.H3("Listening Activity Patterns", style=self._section_title_style()),
                html.Div(
                    "Your listening activity throughout the day. Darker colors indicate more activity.",
                    className='description-box'
                ),
                dcc.Graph(id='temporal-heatmap')
            ], style=self._card_style())
        ])

    def _create_comparison_layout(self):
        return html.Div([
            html.Div([
                html.Div([
                    html.H3("Overall Similarity", style=self._section_title_style()),
                    html.Div(
                        "Percentage score indicating how similar the two users' listening patterns are.",
                        className='description-box'
                    ),
                    dcc.Graph(id='similarity-gauge')
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H3("Genre Comparison", style=self._section_title_style()),
                    html.Div(
                        "Comparison of genre preferences between users.",
                        className='description-box'
                    ),
                    dcc.Graph(id='genre-comparison-radar')
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            
            html.Div([
                html.Div([
                    html.H3("Artist Overlap", style=self._section_title_style()),
                    html.Div(
                        "Visual representation of shared artists between users.",
                        className='description-box'
                    ),
                    dcc.Graph(id='artist-venn')
                ], style={'width': '48%'}),
                
                html.Div([
                    html.H3("Temporal Comparison", style=self._section_title_style()),
                    html.Div(
                        "Comparison of when users typically listen to music.",
                        className='description-box'
                    ),
                    dcc.Graph(id='temporal-comparison')
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'})
        ])

    def _card_style(self):
        return {
            'backgroundColor': self.colors['card_background'],
            'padding': '20px',
            'marginTop': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
        }

    def _section_title_style(self):
        return {
            'color': self.colors['text'],
            'marginBottom': '15px',
            'fontSize': '1.2em',
            'fontFamily': 'monospace',
            'letterSpacing': '1px'
        }

    def _tab_style(self):
        return {
            'padding': '12px',
            'backgroundColor': self.colors['background'],
            'color': self.colors['text'],
            'border': f'1px solid {self.colors["grid"]}',
            'borderRadius': '5px 5px 0 0',
            'marginRight': '2px',
            'fontFamily': 'monospace'
        }

    def _tab_selected_style(self):
        return {
            'padding': '12px',
            'backgroundColor': self.colors['card_background'],
            'color': self.colors['primary'],
            'border': f'1px solid {self.colors["primary"]}',
            'borderBottom': 'none',
            'borderRadius': '5px 5px 0 0',
            'marginRight': '2px',
            'fontFamily': 'monospace'
        }

    def register_callbacks(self):
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
        
        colors = [
            self.colors['primary'],    # Cyan
            self.colors['secondary'],  # Green
            self.colors['accent'],     # Purple
            self.colors['highlight'],  # Pink
            '#FFB86C',                # Orange
            '#FF5555',                # Red
            '#F1FA8C',                # Yellow
            '#50FA7B'                 # Light green
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(
                color=self.colors['text'],
                size=12
            ),
            marker=dict(
                colors=colors * (len(labels)//len(colors) + 1),
                line=dict(color=self.colors['background'], width=2)
            ),
            hovertemplate="<b>%{label}</b><br>" +
                         "Count: %{value}<br>" +
                         "Percentage: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            title=dict(
                text="Genre Distribution",
                font=dict(color=self.colors['text'], size=20)
            ),
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            height=600,  # Increased height
            margin=dict(
                l=50,    # Left margin
                r=120,   # Increased right margin for legend
                t=80,    # Top margin
                b=50     # Bottom margin
            ),
            showlegend=True,
            legend=dict(
                font=dict(
                    color=self.colors['text'],
                    size=12
                ),
                bgcolor='rgba(0,0,0,0)',
                bordercolor=self.colors['text'],
                borderwidth=1,
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1,  # Moved legend further right
                orientation="v"
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
            textfont=dict(color=self.colors['text']),
            hovertemplate="<b>%{x}</b><br>" +
                         "Plays: %{y}<extra></extra>"
        )])
        
        fig.update_layout(
            title=dict(
                text="Top Artists by Play Count",
                font=dict(color=self.colors['text'], size=20)
            ),
            xaxis=dict(
                title="Artist",
                color=self.colors['text'],
                tickfont=dict(color=self.colors['muted_text']),
                showgrid=True,
                gridcolor=self.colors['grid'],
                tickangle=-45
            ),
            yaxis=dict(
                title="Play Count",
                color=self.colors['text'],
                tickfont=dict(color=self.colors['muted_text']),
                showgrid=True,
                gridcolor=self.colors['grid']
            ),
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            height=500,
            margin=dict(l=50, r=50, t=80, b=120),  # Increased bottom margin for rotated labels
            showlegend=False
        )
        return fig

    def _create_temporal_heatmap(self, data: Dict) -> go.Figure:
        temporal = data['temporal_patterns']['distributions']['hourly']
        hours = list(range(24))
        values = [temporal.get(str(hour), 0) for hour in hours]
        
        # Calculate max value for better color scaling
        max_value = max(values) if values else 1
        
        fig = go.Figure(data=go.Heatmap(
            z=[values],
            x=hours,
            colorscale=[
                [0, 'rgba(44, 44, 84, 0.2)'],      # Very dark/transparent for lowest
                [0.25, 'rgba(98, 114, 164, 0.4)'],  # Lighter for low
                [0.5, 'rgba(139, 233, 253, 0.6)'],  # Cyan for medium
                [0.75, 'rgba(189, 147, 249, 0.8)'], # Purple for high
                [1, 'rgba(255, 121, 198, 1)']       # Pink for highest
            ],
            showscale=True,
            hoverongaps=False,
            hovertemplate="Hour: %{x}:00<br>" +
                         "Activity Level: %{z}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(
                text="Hourly Listening Activity",
                font=dict(color=self.colors['text'], size=20)
            ),
            xaxis=dict(
                title="Hour of Day",
                titlefont=dict(color=self.colors['text']),
                ticktext=[f"{hour:02d}:00" for hour in hours],
                tickvals=hours,
                color=self.colors['text'],
                tickfont=dict(color=self.colors['muted_text']),
                gridcolor=self.colors['grid']
            ),
            yaxis=dict(
                showticklabels=False,
                gridcolor=self.colors['grid']
            ),
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            height=300,
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(
                title="Activity Level",
                titlefont=dict(color=self.colors['text'], size=12),
                tickfont=dict(color=self.colors['text']),
                ticktext=['Low', 'Medium', 'High'],
                tickvals=[max_value * x for x in [0.25, 0.5, 0.75]],
                bgcolor=self.colors['card_background'],
                bordercolor=self.colors['text'],
                borderwidth=1,
                thicknessmode="pixels",
                thickness=20,
                len=0.9,
                x=1.02
            )
        )
        return fig

    def _create_metrics_gauge(self, data: Dict) -> go.Figure:
        metrics = data['user_metrics']['listening_stats']
        
        fig = go.Figure()
        
        # Listening Time Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['total_time_hours'],
            domain={'x': [0, 0.45], 'y': [0, 1]},
            title={
                'text': "Listening Time (Hours)",
                'font': {'color': self.colors['text'], 'size': 16}
            },
            number={
                'font': {'color': self.colors['text'], 'size': 24},
                'valueformat': '.1f'
            },
            gauge={
                'axis': {
                    'range': [0, max(100, metrics['total_time_hours'])],
                    'tickfont': {'color': self.colors['muted_text']},
                    'tickcolor': self.colors['text']
                },
                'bar': {'color': self.colors['primary']},
                'bgcolor': self.colors['background'],
                'borderwidth': 2,
                'bordercolor': self.colors['text'],
                'steps': [
                    {'range': [0, max(100, metrics['total_time_hours'])/2], 
                     'color': self.colors['accent']},
                    {'range': [max(100, metrics['total_time_hours'])/2, max(100, metrics['total_time_hours'])], 
                     'color': self.colors['background']}
                ]
            }
        ))
        
        # Unique Tracks Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['unique_tracks'],
            domain={'x': [0.55, 1], 'y': [0, 1]},
            title={
                'text': "Unique Tracks",
                'font': {'color': self.colors['text'], 'size': 16}
            },
            number={
                'font': {'color': self.colors['text'], 'size': 24},
                'valueformat': ','
            },
            gauge={
                'axis': {
                    'range': [0, max(100, metrics['unique_tracks'])],
                    'tickfont': {'color': self.colors['muted_text']},
                    'tickcolor': self.colors['text']
                },
                'bar': {'color': self.colors['secondary']},
                'bgcolor': self.colors['background'],
                'borderwidth': 2,
                'bordercolor': self.colors['text'],
                'steps': [
                    {'range': [0, max(100, metrics['unique_tracks'])/2], 
                     'color': self.colors['accent']},
                    {'range': [max(100, metrics['unique_tracks'])/2, max(100, metrics['unique_tracks'])], 
                     'color': self.colors['background']}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(family='monospace'),
            showlegend=False
        )
        return fig

    def _create_similarity_gauge(self, data: Dict) -> go.Figure:
        similarity = data['overall_similarity']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=similarity,
            title={
                'text': "Overall Similarity Score (%)",
                'font': {'color': self.colors['text'], 'size': 16}
            },
            number={
                'font': {'color': self.colors['text'], 'size': 24},
                'suffix': '%'
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickfont': {'color': self.colors['muted_text']},
                    'tickcolor': self.colors['text']
                },
                'bar': {'color': self.colors['primary']},
                'bgcolor': self.colors['background'],
                'borderwidth': 2,
                'bordercolor': self.colors['text'],
                'steps': [
                    {'range': [0, 33], 'color': self.colors['background']},
                    {'range': [33, 66], 'color': self.colors['accent']},
                    {'range': [66, 100], 'color': self.colors['secondary']}
                ],
                'threshold': {
                    'line': {'color': self.colors['highlight'], 'width': 4},
                    'thickness': 0.75,
                    'value': similarity
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            font=dict(family='monospace')
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
            line_color=self.colors['primary'],
            hovertemplate="Genre: %{theta}<br>" +
                         "Count: %{r}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[genre[2] for genre in shared_genres],
            theta=[genre[0] for genre in shared_genres],
            fill='toself',
            name='User 2',
            line_color=self.colors['secondary'],
            hovertemplate="Genre: %{theta}<br>" +
                         "Count: %{r}<extra></extra>"
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont={'color': self.colors['muted_text']},
                    gridcolor=self.colors['grid']
                ),
                angularaxis=dict(
                    tickfont={'color': self.colors['text']},
                    gridcolor=self.colors['grid']
                ),
                bgcolor=self.colors['background']
            ),
            showlegend=True,
            legend=dict(
                font={'color': self.colors['text']},
                bgcolor='rgba(0,0,0,0)',
                x=0.5,
                y=-0.2,
                xanchor='center',
                orientation='h'
            ),
            height=400,
            margin=dict(l=50, r=50, t=50, b=80),
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            font=dict(family='monospace')
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
        
        # Add annotations with improved styling
        fig.add_annotation(
            x=-0.5, y=0,
            text=f"User 1<br>{user1_unique} artists",
            showarrow=False,
            font=dict(color=self.colors['text'], size=14),
            bgcolor=self.colors['background'],
            bordercolor=self.colors['primary'],
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
        
        fig.add_annotation(
            x=1.5, y=0,
            text=f"User 2<br>{user2_unique} artists",
            showarrow=False,
            font=dict(color=self.colors['text'], size=14),
            bgcolor=self.colors['background'],
            bordercolor=self.colors['secondary'],
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
        
        fig.add_annotation(
            x=0.5, y=0,
            text=f"Common<br>{common_count} artists",
            showarrow=False,
            font=dict(color=self.colors['text'], size=14),
            bgcolor=self.colors['background'],
            bordercolor=self.colors['accent'],
            borderwidth=1,
            borderpad=4,
            opacity=0.8
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
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            font=dict(family='monospace')
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
            textfont={'color': self.colors['text']},
            hovertemplate="Metric: %{x}<br>" +
                         "Similarity: %{y:.1f}%<br>" +
                         "<extra></extra>"
        )])
        
        fig.update_layout(
            title={
                'text': "Temporal Pattern Similarity",
                'font': {'color': self.colors['text'], 'size': 20}
            },
            yaxis=dict(
                title="Similarity (%)",
                range=[0, 100],
                tickfont={'color': self.colors['muted_text']},
                gridcolor=self.colors['grid'],
                titlefont={'color': self.colors['text']}
            ),
            xaxis=dict(
                tickfont={'color': self.colors['muted_text']},
                gridcolor=self.colors['grid']
            ),
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor=self.colors['card_background'],
            plot_bgcolor=self.colors['plot_background'],
            font=dict(family='monospace'),
            showlegend=False,
            bargap=0.3
        )
        return fig

    def run_server(self, debug=False, port=8050, host='localhost', use_reloader=False):
        """Run the dashboard server with error handling"""
        try:
            self.app.run_server(
                debug=debug,
                port=port,
                host=host,
                use_reloader=use_reloader
            )
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run dashboard
    try:
        dashboard = SpotifyDashboard()
        dashboard.run_server(debug=True)
    except Exception as e:
        logging.error(f"Failed to start dashboard: {e}")
        raise
