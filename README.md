# Particle Life Simulation

A high performance Python simulation of artificial life using particles and simple interaction rules. 

<img width="492" height="493" alt="Screenshot 2026-01-06 170456" src="https://github.com/user-attachments/assets/50c519e4-207d-45ac-bbd0-a3988ef93054" />

## Project Overview
This project explores **emergence**: how complex, life like structures can arise from simple local interaction rules. 

Unlike standard implementations that run in $O(n^2)$ time, this simulation utilizes **Spatial Partitioning (Grid Hashing)** and **Numba JIT compilation** to achieve approximately $O(n)$ performance, allowing for tens of thousands of particles to interact in realtime.

## Features
* **High Performance:** Simulates 10,000+ particles in real-time using Numba and Spatial Hashing.
* **Emergent Behavior:** Observe self-organizing structures based on simple attraction/repulsion matrices.
* **Interactive UI:** Adjust parameters using an interactive UI
* **Periodic Boundaries:** The world wraps around (toroidal topology) to prevent edge clumping.

<img width="873" height="534" alt="Screenshot 2026-01-12 115641" src="https://github.com/user-attachments/assets/2e1f6dcc-637d-4960-b905-04c65b7a3373" />

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Alek-Klem-UU/Particle-Life.git
    cd Particle-Life
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    

## Usage

Run the main simulation script:

```bash
python main.py
