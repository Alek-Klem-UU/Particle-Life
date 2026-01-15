# Particle Life Simulation

A high performance Python simulation of artificial life using particles and simple interaction rules. 

<img width="492" height="493" alt="Screenshot 2026-01-06 170456" src="https://github.com/user-attachments/assets/50c519e4-207d-45ac-bbd0-a3988ef93054" />
<img width="499" height="500" alt="Screenshot 2026-01-12 125238" src="https://github.com/user-attachments/assets/d27eebd4-7161-4630-a277-b5afcdfeb9a5" />
<img width="593" height="599" alt="image" src="https://github.com/user-attachments/assets/d77a3bf9-b57e-47b7-80e4-ff0dc0a6ea9d" />


## Project Overview
This project explores **emergence**: how complex, life like structures can arise from simple local interaction rules. 

Unlike standard implementations that run in $O(n^2)$ time, this simulation utilizes **Spatial Partitioning (Grid Hashing)** and **Numba JIT compilation** to achieve approximately $O(n)$ performance, allowing for tens of thousands of particles to interact in realtime.

## Features
* **High Performance:** Simulates 10,000+ particles in real-time using Numba and Spatial Hashing.
* **Emergent Behavior:** Observe self-organizing structures based on simple attraction/repulsion matrices.
* **Interactive UI:** Adjust parameters using an interactive UI
* **Periodic Boundaries:** The world wraps around (toroidal topology) to prevent edge clumping.
* **Live Plot Data:** Keep track of performance live.
* **Cool Seed Options:** Four seeds which are cool in my opinion.

<img width="1093" height="601" alt="Screenshot 2026-01-14 135100" src="https://github.com/user-attachments/assets/14e10f3e-31c6-46ec-bf89-5a72cca5b199" />

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
