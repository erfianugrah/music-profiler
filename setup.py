from setuptools import setup, find_packages

setup(
    name="spotify-profiler",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "spotipy>=2.23.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "dash>=2.9.0",
        "plotly>=5.13.0",
        "python-dotenv>=0.20.0",
    ],
    author="Erfi Anugrah",
    description="A tool for analyzing Spotify listening history",
    python_requires=">=3.8",
)
