import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class Node:
    def __init__(self, node_id, x, y, bc=False, temperature=None):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.bc = bc
        self.temperature = temperature  # stationary

    def __repr__(self):
        return f"Node({self.node_id}, x={self.x}, y={self.y}, bc={self.bc})"


class Element:
    def __init__(self, element_id, nodes):
        self.element_id = element_id
        self.nodes = nodes
        self.H = np.zeros((4, 4))
        self.Hbc = np.zeros((4, 4))
        self.P = np.zeros(4)
        self.C = np.zeros((4, 4))

    def __repr__(self):
        return f"Element({self.element_id}, nodes={self.nodes})"


class Grid:
    def __init__(self):
        self.nodes = []  # list of Node objects
        self.elements = []  # list of Element objects
        self.H = np.zeros((len(self.nodes), len(self.nodes)))
        self.Hbc = np.zeros((len(self.nodes), len(self.nodes)))
        self.P = np.zeros(len(self.nodes))
        self.C = np.zeros((len(self.nodes), len(self.nodes)))

    def __repr__(self):
        return f"Grid(nN={len(self.nodes)}, nE={len(self.elements)})"


class GlobalData:
    def __init__(self, simulation_time, step_time, conductivity, alfa, tot, initial_temp, density, specific_heat, n_nodes, n_elements, height, width):
        self.simulation_time = simulation_time
        self.step_time = step_time
        self.conductivity = conductivity
        self.alfa = alfa
        self.tot = tot
        self.initial_temp = initial_temp
        self.density = density
        self.specific_heat = specific_heat
        self.n_nodes = n_nodes
        self.n_elements = n_elements
        self.height = height
        self.width = width

    def __repr__(self):
        return f"GlobalData(simulation_time={self.simulation_time}, step_time={self.step_time}, ... nN={self.n_nodes}, nE={self.n_elements})"


def parse_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    global_data = None
    grid = Grid()
    reading_nodes = False
    reading_elements = False
    reading_bc = False  # Flag for *BC section
    boundary_conditions = set()

    for line in lines:
        line = line.strip()

        if line.startswith("SimulationTime"):
            simulation_time = float(line.split()[1])
        elif line.startswith("SimulationStepTime"):
            step_time = float(line.split()[1])
        elif line.startswith("Conductivity"):
            conductivity = float(line.split()[1])
        elif line.startswith("Alfa"):
            alfa = float(line.split()[1])
        elif line.startswith("Tot"):
            tot = float(line.split()[1])
        elif line.startswith("InitialTemp"):
            initial_temp = float(line.split()[1])
        elif line.startswith("Density"):
            density = float(line.split()[1])
        elif line.startswith("SpecificHeat"):
            specific_heat = float(line.split()[1])
        elif line.startswith("Nodes number"):
            n_nodes = int(line.split()[2])
        elif line.startswith("Elements number"):
            n_elements = int(line.split()[2])
        elif line.startswith("*Node"):
            reading_nodes = True
            reading_elements = False
            reading_bc = False
        elif line.startswith("*Element"):
            reading_nodes = False
            reading_elements = True
            reading_bc = False
        elif line.startswith("*BC"):
            reading_nodes = False
            reading_elements = False
            reading_bc = True
        elif reading_bc and line:  # Reading *BC section
            bc_nodes = map(int, line.split(','))
            boundary_conditions.update(bc_nodes)
        elif reading_nodes and line:
            parts = line.split(',')
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            grid.nodes.append(Node(node_id, x, y))
        elif reading_elements and line:
            parts = line.split(',')
            element_id = int(parts[0])
            nodes = list(map(int, parts[1:]))
            grid.elements.append(Element(element_id, nodes))

    # Update boundary condition for nodes
    for node in grid.nodes:
        if node.node_id in boundary_conditions:
            node.bc = True

    global_data = GlobalData(simulation_time, step_time, conductivity, alfa, tot, initial_temp, density, specific_heat, n_nodes, n_elements, 0, 0)
    return global_data, grid


def plot_grid(grid):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.8)  # Leave space for buttons on the right

    # Plot elements and nodes
    for element in grid.elements:
        x_coords = [grid.nodes[node_id - 1].x for node_id in element.nodes]
        y_coords = [grid.nodes[node_id - 1].y for node_id in element.nodes]
        x_coords.append(grid.nodes[element.nodes[0] - 1].x)  # Close the element loop
        y_coords.append(grid.nodes[element.nodes[0] - 1].y)
        ax.plot(x_coords, y_coords, 'b-')

    node_labels = {}
    for node in grid.nodes:
        ax.plot(node.x, node.y, 'ro')
        node_labels[node.node_id] = ax.text(node.x, node.y, str(node.node_id), color="red", fontsize=12, visible=False)

    element_labels = {}
    for element in grid.elements:
        centroid_x = sum(grid.nodes[node_id - 1].x for node_id in element.nodes) / len(element.nodes)
        centroid_y = sum(grid.nodes[node_id - 1].y for node_id in element.nodes) / len(element.nodes)
        element_labels[element.element_id] = ax.text(centroid_x, centroid_y, f"E{element.element_id}", color="blue", fontsize=12, visible=False)

    def toggle_node_labels(event):
        visibility = not node_labels[next(iter(node_labels))].get_visible()
        for label in node_labels.values():
            label.set_visible(visibility)
        plt.draw()

    def toggle_element_labels(event):
        visibility = not element_labels[next(iter(element_labels))].get_visible()
        for label in element_labels.values():
            label.set_visible(visibility)
        plt.draw()

    ax_toggle_nodes = plt.axes([0.82, 0.6, 0.15, 0.075])
    btn_toggle_nodes = Button(ax_toggle_nodes, 'Nodes')
    btn_toggle_nodes.on_clicked(toggle_node_labels)

    ax_toggle_elements = plt.axes([0.82, 0.5, 0.15, 0.075])
    btn_toggle_elements = Button(ax_toggle_elements, 'Elements')
    btn_toggle_elements.on_clicked(toggle_element_labels)

    ax.set_xlabel("X Coordinate", labelpad=20)
    ax.set_ylabel("Y Coordinate", labelpad=20)
    ax.set_title("Finite Element Grid", pad=20)
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()


def print_matrix(matrix):
    for row in matrix:
        print("\t".join(map(str, row)))


def initialize_integration_points(n):
    if n == 2:
        return [-np.sqrt(1 / 3), np.sqrt(1 / 3)], [1, 1]
    elif n == 3:
        return [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)], [5 / 9, 8 / 9, 5 / 9]
    elif n == 4:
        return [
            -np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
            -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
            np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
            np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
        ], [
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ]
    else:
        raise ValueError("Invalid integration points")

def calculate_shape_function_derivatives(ksi, eta):
    dN_dksi = [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)]
    dN_deta = [-0.25 * (1 - ksi), -0.25 * (1 + ksi), 0.25 * (1 + ksi), 0.25 * (1 - ksi)]
    return dN_dksi, dN_deta

def calculate_jacobian(dN_dksi, dN_deta, nodes):
    x = [node.x for node in nodes]
    y = [node.y for node in nodes]
    J_11 = sum(dN_dksi[i] * x[i] for i in range(4))
    J_12 = sum(dN_dksi[i] * y[i] for i in range(4))
    J_21 = sum(dN_deta[i] * x[i] for i in range(4))
    J_22 = sum(dN_deta[i] * y[i] for i in range(4))
    J = [[J_11, J_12], [J_21, J_22]]
    det_J = J_11 * J_22 - J_12 * J_21
    if det_J == 0:
        raise ValueError("Jacobian is singular")
    return J, det_J


def calculate_jacobian_inverse(J, det_J):
    J_11, J_12 = J[0]
    J_21, J_22 = J[1]
    invJ = [[ J_22 / det_J, -J_12 / det_J], [-J_21 / det_J,  J_11 / det_J]]
    return invJ

def calculate_xy_derivatives(invJ, dN_dksi, dN_deta):
    dN_dx = [invJ[0][0] * dN_dksi[i] + invJ[0][1] * dN_deta[i] for i in range(4)]
    dN_dy = [invJ[1][0] * dN_dksi[i] + invJ[1][1] * dN_deta[i] for i in range(4)]
    return dN_dx, dN_dy

def calculate_H_matrix_pc(dN_dx, dN_dy, det_J, conductivity):
    Hpc = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            Hpc[a][b] = (dN_dx[a] * dN_dx[b] + dN_dy[a] * dN_dy[b]) * conductivity * det_J
    return Hpc


def calculate_H_matrix_for_element(element, grid, conductivity, n):
    nodes = [grid.nodes[node_id - 1] for node_id in element.nodes]
    p, w = initialize_integration_points(n)
    H = np.zeros((4, 4))
    for i in range(len(p)):
        for j in range(len(p)):
            eta, ksi = p[i], p[j]
            #print(f"\nIntegration point: ksi={ksi}, eta={eta}")

            # Calculate shape function derivatives with respect to ksi and eta
            dN_dksi, dN_deta = calculate_shape_function_derivatives(ksi, eta)
            #print(f"Shape function derivatives with respect to ksi: {dN_dksi}")
            #print(f"Shape function derivatives with respect to eta: {dN_deta}")

            # Calculate Jacobian and its determinant
            J, det_J = calculate_jacobian(dN_dksi, dN_deta, nodes)
            #print(f"Jacobian J: {J}")
            # Calculate Jacobian inverse
            invJ = calculate_jacobian_inverse(J, det_J)
            #print(f"Jacobian inverse: {invJ}")

            # Calculate derivatives with respect to x and y
            dN_dx, dN_dy = calculate_xy_derivatives(invJ, dN_dksi, dN_deta)
            #print(f"Shape function derivatives with respect to x: {dN_dx}")
            #print(f"Shape function derivatives with respect to y: {dN_dy}")

            # Calculate Hpc matrix for this integration point
            Hpc = calculate_H_matrix_pc(dN_dx, dN_dy, det_J, conductivity)
            #print(f"Hpc matrix for integration point ({ksi}, {eta}):")
            #print(Hpc)

            # Add to H matrix
            H += Hpc

    element.H = H
    print(f"H matrix for element {element.element_id}:")
    print(element.H)

def calculate_Hbc_pc(N, alfa, det_J, weight):
    Hbc_pc = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            Hbc_pc[a][b] = alfa * N[a] * N[b] * det_J * weight
    return Hbc_pc


def calculate_Hbc_for_edge(node1, node2, alfa, p, w):
    Hbc_edge = np.zeros((2, 2))

    # Edge length
    edge_length = np.sqrt((node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2)
    det_J = edge_length / 2  # Jacobian determinant for linear edge

    # Iterate over integration points
    for i in range(len(p)):
        ksi = p[i]
        weight = w[i]

        # Shape functions for line (edge)
        N = [(1 - ksi) / 2, (1 + ksi) / 2]

        # Calculate Hbc for integration point
        Hbc_pc = calculate_Hbc_pc(N, alfa, det_J, weight)
        Hbc_edge += Hbc_pc

    return Hbc_edge


def calculate_Hbc_for_element(element, grid, alfa, n):
    nodes = [grid.nodes[node_id - 1] for node_id in element.nodes]
    p, w = initialize_integration_points(n)
    Hbc = np.zeros((4, 4))

    # List of element edges: (start_node, end_node)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for edge in edges:
        node1, node2 = nodes[edge[0]], nodes[edge[1]]

        # Check if both nodes are on the boundary
        if not (node1.bc and node2.bc):
            continue

        # Calculate local Hbc matrix for edge
        Hbc_edge = calculate_Hbc_for_edge(node1, node2, alfa, p, w)

        # Map local values to global positions in element matrix
        for a in range(2):
            for b in range(2):
                global_a = edge[a]
                global_b = edge[b]
                Hbc[global_a][global_b] += Hbc_edge[a][b]

    element.Hbc = Hbc
    print(f"Hbc matrix for element {element.element_id}:\n", Hbc)


def calculate_P_vector_pc(N, alfa, T_ot, det_J, weight):
    P_pc = np.zeros(2)
    for a in range(2):
        P_pc[a] = alfa * T_ot * N[a] * det_J * weight
    return P_pc


def calculate_P_vector_for_edge(node1, node2, alfa, T_ot, p, w):
    P_edge = np.zeros(2)

    # Edge length
    edge_length = np.sqrt((node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2)
    det_J = edge_length / 2  # Jacobian determinant for linear edge

    # Iterate over integration points
    for i in range(len(p)):
        ksi = p[i]
        weight = w[i]

        # Shape functions for line (edge)
        N = [(1 - ksi) / 2, (1 + ksi) / 2]

        # Calculate P for integration point
        P_pc = calculate_P_vector_pc(N, alfa, T_ot, det_J, weight)
        P_edge += P_pc

    return P_edge


def calculate_P_vector_for_element(element, grid, alfa, T_ot, n):
    nodes = [grid.nodes[node_id - 1] for node_id in element.nodes]
    p, w = initialize_integration_points(n)
    P = np.zeros(4)

    # List of element edges: (start_node, end_node)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for edge in edges:
        node1, node2 = nodes[edge[0]], nodes[edge[1]]

        # Check if both nodes are on the boundary
        if not (node1.bc and node2.bc):
            continue

        # Calculate local P vector for edge
        P_edge = calculate_P_vector_for_edge(node1, node2, alfa, T_ot, p, w)

        # Map local values to global positions in element vector
        for a in range(2):
            global_a = edge[a]
            P[global_a] += P_edge[a]

    element.P = P
    print(f"P vector for element {element.element_id}:\n", P)



def calculate_C_matrix_pc(N, det_J, density, specific_heat):
    Cpc = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            Cpc[a][b] = density * specific_heat * N[a] * N[b] * det_J
    return Cpc


def calculate_C_matrix_for_element(element, grid, density, specific_heat, n):
    nodes = [grid.nodes[node_id - 1] for node_id in element.nodes]
    p, w = initialize_integration_points(n)
    C = np.zeros((4, 4))

    for i in range(len(p)):
        for j in range(len(p)):
            ksi, eta = p[i], p[j]
            weight = w[i] * w[j]

            # Shape functions at given integration point
            N = [
                0.25 * (1 - ksi) * (1 - eta),
                0.25 * (1 + ksi) * (1 - eta),
                0.25 * (1 + ksi) * (1 + eta),
                0.25 * (1 - ksi) * (1 + eta),
            ]

            # Calculate shape function derivatives and Jacobian
            dN_dksi, dN_deta = calculate_shape_function_derivatives(ksi, eta)
            J, det_J = calculate_jacobian(dN_dksi, dN_deta, nodes)

            if det_J <= 0:
                raise ValueError("Invalid Jacobian determinant: det(J) <= 0.")

            # Calculate local C matrix for integration point
            Cpc = calculate_C_matrix_pc(N, det_J, density, specific_heat)
            C += Cpc * weight

    element.C = C
    print(f"C matrix for element {element.element_id}:\n", C)


def calculate_global_H_matrix(grid):
    n_nodes = len(grid.nodes)
    H_global = np.zeros((n_nodes, n_nodes))

    for element in grid.elements:
        for i in range(4):
            for j in range(4):
                global_i = element.nodes[i] - 1
                global_j = element.nodes[j] - 1
                H_global[global_i][global_j] += element.H[i][j]

    grid.H = H_global
    return H_global

def calculate_global_Hbc_matrix(grid):
    n_nodes = len(grid.nodes)
    Hbc_global = np.zeros((n_nodes, n_nodes))

    for element in grid.elements:
        for i in range(4):
            for j in range(4):
                global_i = element.nodes[i] - 1
                global_j = element.nodes[j] - 1
                Hbc_global[global_i][global_j] += element.Hbc[i][j]

    grid.Hbc = Hbc_global
    return Hbc_global

def calculate_global_P_vector(grid):
    n_nodes = len(grid.nodes)
    P_global = np.zeros(n_nodes)

    for element in grid.elements:
        for i in range(4):
            global_i = element.nodes[i] - 1
            P_global[global_i] += element.P[i]

    grid.P = P_global
    return P_global


def calculate_global_C_matrix(grid):
    n_nodes = len(grid.nodes)
    C_global = np.zeros((n_nodes, n_nodes))

    for element in grid.elements:
        for i in range(4):
            for j in range(4):
                global_i = element.nodes[i] - 1
                global_j = element.nodes[j] - 1
                C_global[global_i][global_j] += element.C[i][j]

    grid.C = C_global
    return C_global


def gaussian_elimination(A, b):
    A = A.tolist()  # Convert matrix A to lists
    b = b.tolist()  # Convert vector b to lists
    n = len(A)

    # Create augmented matrix
    for i in range(n):
        A[i].append(b[i])

    # Forward elimination
    for i in range(n):
        # Pivot selection
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]

        # Ensure pivot is non-zero
        if abs(A[i][i]) < 1e-12:
            raise ValueError("Matrix is singular or system has no unique solution.")

        # Eliminate lower rows
        for j in range(i + 1, n):
            ratio = A[j][i] / A[i][i]
            for k in range(i, n + 1):
                A[j][k] -= ratio * A[i][k]

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][-1]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x


def solve_stationary(grid):
    # Solves the stationary system of equations {H}t + {P} = 0.

    # Sum H_global and Hbc_global matrices
    H_combined = grid.H + grid.Hbc

    # Right-hand side of equation
    P_combined = grid.P

    # Solve system of equations H_combined * t = P_combined
    t = gaussian_elimination(H_combined, P_combined)
    for i, node in enumerate(grid.nodes):
        node.temperature = t[i]
    print("\nStationary solution (temperature vector):")
    print(t)
    return t


def solve_transient(global_data, grid):
    # Solves the transient system of equations {H}[t] + {P} + [C](d[t]/dτ) = 0 iteratively for time steps.

    # Global matrices
    H_global = grid.H + grid.Hbc  # Global H matrix including boundary conditions
    C_global = grid.C  # Global C matrix
    P_global = grid.P  # Global P vector

    # Initial conditions
    t_current = np.full(len(grid.nodes), global_data.initial_temp)

    # Time and time step
    delta_tau = global_data.step_time
    total_time = global_data.simulation_time

    # Store results
    print("\nStarting transient problem simulation:")
    max_temp = float('-inf')
    min_temp = float('inf')

    for current_time in np.arange(delta_tau, total_time + delta_tau, delta_tau):
        # Prepare matrices for time step
        A = H_global + (C_global / delta_tau)  # Left-hand side matrix
        b = P_global + (C_global @ t_current / delta_tau)  # Right-hand side vector

        # Solve system of linear equations for t_next
        t_next = gaussian_elimination(A, b)

        # Update temperature and time
        t_current = t_next

        # Print results for current time
        #print(f"Time: {current_time}")
        print(f"MinTemp: {np.min(t_current)}, MaxTemp: {np.max(t_current)}")


def FEM(file_path, n_integration_points=2):

    # Parse file
    global_data, grid = parse_file(file_path)
    print(global_data)

    # Display grid
    plot_grid(grid)

    # Calculations for elements
    for element in grid.elements:
        print(f"\n\nCALCULATIONS FOR ELEMENT {element.element_id}")

        # Calculate local H matrix
        calculate_H_matrix_for_element(element, grid, global_data.conductivity, n_integration_points)

        # Calculate local Hbc matrix
        calculate_Hbc_for_element(element, grid, global_data.alfa, n_integration_points)

        # Calculate local P vector
        calculate_P_vector_for_element(element, grid, global_data.alfa, global_data.tot, n_integration_points)

        # Calculate local C matrix (added)
        calculate_C_matrix_for_element(element, grid, global_data.density, global_data.specific_heat, n_integration_points)

    # Calculate global matrices and vector
    H_global = calculate_global_H_matrix(grid)
    Hbc_global = calculate_global_Hbc_matrix(grid)
    P_global = calculate_global_P_vector(grid)
    C_global = calculate_global_C_matrix(grid)

    print("\nGlobal H matrix:")
    print_matrix(H_global)

    print("\nGlobal Hbc matrix:")
    print_matrix(Hbc_global)

    print("\nGlobal P vector:")
    print(P_global)

    print("\nGlobal C matrix:")
    print_matrix(C_global)

    # Solve stationary system of equations
    t = solve_stationary(grid)

    # Solve transient problem
    solve_transient(global_data, grid)


# Run FEM function for input file
results = FEM('example_input.txt')