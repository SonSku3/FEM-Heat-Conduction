# Finite Element Method (FEM) Heat Conduction Simulation

This project implements a 2D **Finite Element Method (FEM)** solver for stationary and transient **heat conduction** problems.  
It was originally developed as part of a university project.

---

## Project Overview

The program reads mesh and material data from a structured text file and computes:
- Local and global FEM matrices (`H`, `Hbc`, `C`, and `P`),
- Stationary and transient temperature distributions,
- Simple mesh visualization (nodes, elements, boundary conditions).

It demonstrates a full workflow of a simple FEM solver — from data parsing and numerical integration (Gaussian quadrature) to visualization and solving linear systems.

---

## Features

- File parser for mesh and simulation parameters  
- Node/element classes with OOP design  
- Element-level FEM matrix computation  
- Global matrix assembly  
- Stationary and transient problem solvers  
- Gaussian elimination implementation  
- Interactive mesh plot using Matplotlib

---

## Example Input File (`example_input.txt`)

> Due to copyright restrictions, original university test files are **not included**.  
> Repository contains an example file `example_input.txt` as replacement for original files:


---

## Usage

1. Place your mesh file (e.g., `example_input.txt`) in the same folder as `mes.py`.  
2. Run the solver.

### This will:
- Parse the input file,
- Visualize the mesh,
- Compute all element and global matrices,
- Display steady-state and transient temperature results.

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies using:

```
pip install numpy
pip install matplotlib

```
