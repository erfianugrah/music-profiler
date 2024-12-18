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
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "urllib3>=2.0.0",
        "musicbrainzngs>=0.7.1",
        "scipy>=1.10.0",
        "pytest>=7.3.1",
        "concurrent-futures>=3.0.5",
        "dash-bootstrap-components>=1.0.0"
    ],
    author="Erfi Anugrah",
    author_email="your.email@example.com",
    description="A tool for analyzing Spotify listening history with enhanced metadata",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="spotify, music, analysis, data, musicbrainz",
    url="https://github.com/yourusername/spotify-profiler",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "spotify-profiler=spotify_profiler.run_analysis:main",
        ],
    },
)
