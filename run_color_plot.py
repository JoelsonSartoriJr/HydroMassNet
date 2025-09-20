#!/usr/bin/env python3
"""Generate only the color-stellar mass diagram"""

import yaml
from src.hydromassnet.plotting import plot_color_stellar_mass_diagram

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate the plot
plot_color_stellar_mass_diagram(config)
print("Color-stellar mass diagram generated successfully!")
