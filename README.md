# BB Trajectory Simulator

An interactive airsoft BB trajectory simulator that models physical forces including drag, Magnus effect, and Coriolis effect.

## Features

- Interactive trajectory visualization with adjustable parameters
- Real-time trajectory calculations considering:
  - Drag force with Reynolds number adjustment
  - Magnus effect (backspin)
  - Spin decay
  - Coriolis effect
  - Wind effects
- Tolerance analysis for BB weight and diameter variations
- Energy and velocity visualization
- Optimal shooting zone indication

## Requirements

- Python 3.7+
- Required packages:
  - matplotlib
  - numpy
  - pandas
  - seaborn

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the simulator:

   ```bash
   python3 sim-speed.py
   ```