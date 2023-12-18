import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from itertools import combinations, product
import pandas as pd


# ===== Particle operation functions =====
def get_energy_force_2_particles(p_i, p_j, coulomb_constant=1):
    """
    Calculate the electrostatic force between two particles and
    calculate the potential energy between two particles based on Coulomb's Law.

    Args:
        p_i (tuple): Position (x, y) of the first particle.
        p_j (tuple): Position (x, y) of the second particle.
        coulomb_constant (float, optional): Coulomb's constant, default is 1 for normalized calculations.

    Returns:
        numpy.ndarray: The 2D vector representing the force exerted on the first particle by the second.
        float: The scalar value of the potential energy between the two particles.
    """

    p_i, p_j = np.array(p_i), np.array(p_j)
    r_i_j = p_i - p_j
    dist_i_j = np.linalg.norm(r_i_j)
    
    # Avoid division by zero by imposing a minimum distance
    min_dist = 1e-15  # A small number to prevent division by zero
    dist_i_j = max(dist_i_j, min_dist)
    
    # Calculate the force vector using Coulomb's Law
    force_i_j = coulomb_constant * r_i_j / (dist_i_j**3)

    # Calculate the potential energy using Coulomb's Law
    energy_i_j = coulomb_constant / dist_i_j

    return force_i_j, energy_i_j


def get_energy_forces_total(particles):
    """
    Calculate the total potential energy of a system of particles.
    Sum all the force vectors acting on each particle
    
    Args:
        particles (list or numpy.ndarray): A list of tuples or an n x 2 matrix representing the positions of n particles.

    Returns:
        float: The total potential energy of the system.
    """

    # Convert list of tuples to numpy array if it isn't already
    if not isinstance(particles, np.ndarray):
        particles = np.array([list(row) for row in particles])

    # Initialize array to store energies for each particle
    energies = np.zeros(len(particles))

    # Initialize an array to store the net force on each particle
    net_forces = np.zeros_like(particles)  # Assuming 2D particles

    # Calculate energy between each unique pair of particles
    for p1, p2 in combinations(enumerate(particles), 2):
        force_vec, energy_pair = get_energy_force_2_particles(p1[1], p2[1])
        energies[p1[0]] += energy_pair
        energies[p2[0]] = energy_pair
        net_forces[p1[0]] += force_vec
        net_forces[p2[0]] -= force_vec
    
    # Convert list of energies to numpy array and sum them to get total energy
    # total_energy = np.sum(np.array(energies))

    return energies, net_forces


def in_circle(position, radius):
    """Check if a position is within a circle of given radius centered at the origin.
    
    Args:
        position (list or numpy.ndarray): A single positional 2D vector
        radius (float): The radius of the circle
    Returns:
        bool: A True tag if the point is inside the circle, False otherwise
    """
    
    position = np.array(position)
    return (position[0]**2 + position[1]**2) <= radius**2


def initialise_positions_random(n_particles, radius):
    """Generates a uniform random array of particles within a circle with predefined radius

    Args:
        n_particles (int): number of particles to generate
        radius (float): The radius of the circle
    Returns:
        2D positions of particles
    """

    positions = []
    while len(positions) < n_particles:
        x, y = np.random.uniform(-radius*2, radius*2, size=2)
        position = x,y
        if in_circle(position, radius):
            positions.append([x, y])
    positions = np.array(positions)
    return np.array(positions)


# ===== Particle update functions =====
def initialise_particle_dict_random(n_particles, radius):
    """Initialises random particle positions, calculates and assigns
    the force vectors and energies per particle and stores the
    positions, forces and energies in a dictionary.

    Args:
        n_particles (int): number of particles to generate
        radius (float): The radius of the circle
    Returns:
        dict: An object containing the positions, forces and energies of all particles
    """

    # Generate initial positions
    positions = initialise_positions_random(n_particles, radius)
    
    # Calculate forces on each particle
    energies, forces = get_energy_forces_total(positions)

    # Combine positions, forces, and energies into a single dictionary
    particle_dict = {'positions': positions, 'energies': energies, 'forces': forces}

    return particle_dict


def update_particle_dict(particle_dict, pos, i):
    """Updates the particle dictionary based on new position at a specific index

    Args:
        particle_dict (dict): An object containing the positions, forces and energies of all particles
        pos (numpy.ndarray): The new position of the particle
        i (int): The index of the particle to update
    """

    particle_dict['positions'][i] = pos
    energies, forces = get_energy_forces_total(particle_dict['positions'])
    particle_dict['energies'] = energies
    particle_dict['forces'] = forces


def limit_displacement_to_circle_bnd(pts, vecs, rad):
    """Calculates the maximum displacements of a numpy array of points (pts)
    along a numpy array of vectors (vec) before it crosses the boundary of
    a circle with radius (rad). The projection of each point on the circle
    is determined by the solution of the equation (px + a*vx)^2 + (py + a*vy)^2 = r^2
    where px and py are the coordinates of the point, vx and vy are the
    coordinates of the vector and r is the circle radius.
    a is the distance between the point and its projected image and is returned by the function
    """

    vecs_unit = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]

    term_a = vecs_unit[:,0]**2 + vecs_unit[:,1]**2
    term_b = 2 * pts[:,0] * vecs_unit[:,0] * pts[:,1] * vecs_unit[:,1]
    term_c = (pts[:,0] * vecs_unit[:,1])**2 + (pts[:,1] * vecs_unit[:,0])**2
    sqrt_term = np.sqrt(term_a * (rad**2) + term_b - term_c)
    dot = pts[:,0] * vecs_unit[:,0] + pts[:,1] * vecs_unit[:,1]
    intersection_distances = np.where(term_a > 0, np.vstack([(-sqrt_term - dot)/term_a, (sqrt_term - dot)/term_a]), np.zeros((2, term_a.shape[0])))
    # Pick positive (along vector)
    proj_dist = np.max(intersection_distances, axis=0)

    return proj_dist


def reflect_at_circle_bounds(pts, vecs, rad):
    """Calculates displacements of a numpy array of points (pts)
    along a numpy array of vectors (vec) if they "bounce off" the boundary of
    a circle with radius (rad). The projection of each point on the circle
    is determined by the solution of the equation (px + a*vx)^2 + (py + a*vy)^2 = r^2
    where px and py are the coordinates of the point, vx and vy are the
    coordinates of the vector and r is the circle radius.
    a is the distance between the point and its projected image
    """

    proj_dist = limit_displacement_to_circle_bnd(pts, vecs, rad)

    # Scale projection vector
    proj_vecs = np.array(vecs)
    vec_norms = np.linalg.norm(proj_vecs, axis=1)
    vec_mag = np.min(np.vstack((vec_norms, proj_dist)), axis=0)
    norm_residuals = vec_norms - proj_dist
    vec_norms = np.reshape(vec_norms, (vec_norms.shape[0], 1))
    vec_mag = np.reshape(vec_mag, (vec_mag.shape[0], 1))
    proj_vecs /= vec_norms
    proj_vecs *= vec_mag
    proj_pts = pts + proj_vecs

    # Limit norm residuals to not be larger than projection length
    norm_residuals = np.where(norm_residuals <= proj_dist, norm_residuals, proj_dist)

    # Find components of projection vector along radial vector of intersection point
    parallel_coords = np.array([np.dot(proj_pts[i], proj_vecs[i]) for i in range(proj_pts.shape[0])])
    parallel_comps = parallel_coords[:, np.newaxis] * proj_pts 
    ortho_comps = proj_vecs - parallel_comps
    reflect_vecs = - parallel_comps + ortho_comps
    reflect_norms = np.linalg.norm(reflect_vecs, axis=1)
    reflect_vecs /= reflect_norms[:, np.newaxis]
    reflect_vecs *= norm_residuals[:, np.newaxis]
    
    # Reflect points if they are hitting the boundary
    reflect_pts = np.where(norm_residuals[:, np.newaxis] > 0, proj_pts + reflect_vecs, proj_pts)

    return reflect_pts


def move_particle(particle_dict, radius=1, movement_scaler=1, move_mode = 'repell'):
    """Takes an array of particle dictionaries and displaces the
    particle positions along the particle velocity vectors.
    Updates forces and energies once the displacement is done
    """
    
    new_positions = np.array(particle_dict['positions'])

    if move_mode == 'random cartesian':  # Random cartesian coordinates (hit and miss) within circle

        # Determine target position by randomly cartesian coordinates within enclosing square
        # Accept the ones within the circle
        for i in range(particle_dict['positions'].shape[0]):
            while True:
                x_new, y_new = np.random.uniform(-radius, radius, size=2)
                if in_circle((x_new, y_new), radius):
                    break

            pos_new = np.array([x_new, y_new])

            # Scale perturbation vector by movement scaler
            delta_vec = (pos_new - particle_dict['positions'][i]) * movement_scaler

            new_positions[i] += delta_vec

    elif move_mode == 'random polar absolute':  # Random polar coordinates within circle

        # Determine target position by randomly sampling polar coordinates within global circle
        theta_new = np.random.uniform(0, 2*np.pi, particle_dict['positions'].shape[0])
        r_new = np.random.uniform(0, radius, particle_dict['positions'].shape[0])

        x_new = r_new*np.cos(theta_new)
        y_new =  r_new*np.sin(theta_new)

        pos_new = np.vstack((x_new, y_new)).T

        # Scale perturbation vector by movement scaler
        delta_vec = (pos_new - particle_dict['positions']) * movement_scaler

        particle_dict['positions'] += delta_vec
    
    elif move_mode == 'random polar relative':  # Random polar coordinates relative to current position
        
        # Create random polar perturbation vector around current position
        delta_theta = np.random.uniform(0, 2*np.pi, particle_dict['positions'].shape[0])
        delta_r = np.random.uniform(0, radius * 2, particle_dict['positions'].shape[0])
        
        # Construct perturbation vector and scale by movement scaler
        delta_x = delta_r*np.cos(delta_theta)
        delta_y =  delta_r*np.sin(delta_theta)
        delta_vec = np.vstack((delta_x, delta_y)).T * movement_scaler

        # particle_dict['positions'] += delta_vec
        new_positions = reflect_at_circle_bounds(particle_dict['positions'], particle_dict['forces'] * movement_scaler, radius)

    elif move_mode == 'repell':

        # Normalise force vectors
        force_vecs = particle_dict['forces'] / np.linalg.norm(particle_dict['forces'], axis=1)[:, np.newaxis]

        new_positions = reflect_at_circle_bounds(particle_dict['positions'], particle_dict['forces'] * movement_scaler, radius)

    else:
        raise ValueError("move_mode must be one of 'random cartesian', 'random polar absolute', 'random polar relative'")

    return new_positions


def particle_dict_element(particle_dict, i):
    """Returns a dictionary containing the position, energy and force of a single particle
    """
    selection_dict = {}
    for keys in particle_dict.keys():
        selection_dict[keys] = np.array([particle_dict[keys][i]])
    return selection_dict


# ===== Particle visualisation functions =====
def visualise_particles(particle_dict, radius=1, movement_scaler=1, title="Particle positions and velocities", ax=None):
    """Creates a plot visualising the particle positions and velocities
    """

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Visualise particle positions
    coords = particle_dict['positions'].T
    ax.scatter(coords[0], coords[1], s=10, c=particle_dict['energies'])

    # Visualise particle forces
    # ax.quiver(coords[0], coords[1], force_vecs[0], force_vecs[1], particle_dict['energies'], width=0.005, angles='xy')
    for i in range(particle_dict['forces'].shape[0]):
        pos = particle_dict['positions'][i]
        dir = movement_scaler * particle_dict['forces'][i]
        ax.arrow(pos[0], pos[1], dir[0], dir[1], width=0.005, color='lightgrey')
        ax.annotate(i, pos)

    # Create a circle patch with the same radius as used for the position generation
    circle = patches.Circle((0, 0), radius, fill=False, edgecolor='gray', linestyle='-')
    ax.add_patch(circle)

    # Set aspect of the plot to be equal, so the circle isn't skewed
    ax.set_aspect('equal', adjustable='box')

    # Limit axis ranges
    ax.set_xlim((-1.25*radius, 1.25*radius))
    ax.set_ylim((-1.25*radius, 1.25*radius))

    # Setting labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    fig.suptitle(title)

    if not ax:
        # Show the plot
        plt.show()


# ===== Cooling schedule functions =====
def logarithmic_decay_cooling(T_init, t, a, b):
    """https://canvas.uva.nl/courses/39303/pages/lecture-09-the-gibbs-sampler-and-simulated-annealing?module_item_id=1830823 slide 14"""
    T_n = a/(np.log(t+b))
    return T_n

def exponential_decay_cooling(T_init, t, a, b):
    return T_init * (a ** t)


# ====== Simulated annealing function ======
def sim_annealing_move_particles(particle_dict, time_range, radius, T_init=1, cooling_function=exponential_decay_cooling, a=100, b=1, movement_func=move_particle, sort_mode='normal', move_mode='random', movement_scaler=1):
    """
    Perform simulated annealing to update particle positions within a specified radius.
    Each iteration in the time_range represents a single particle movement.
    """

    if cooling_function is None:
        raise ValueError("cooling_function must be provided")

    T = T_init
    total_energy_over_time = np.zeros(time_range)
    positions_over_time = np.zeros((time_range + 1, particle_dict['positions'].shape[0], 2))

    mc_index = 0 # Markov chain index
    for t in range(time_range):
        
        # Optionally decrease the movement scaler over time for 'incremental drop-off' mode
        if move_mode == 'incremental drop-off':
            movement_scaler /= (1 + t / time_range)
        
        # At each full Markov chain cycle, sort indices of particles to move based on the mode
        if t % particle_dict['positions'].shape[0] == 0:
            mc_index = 0
            if sort_mode == 'random':
                particle_indices = np.random.permutation(np.arange(particle_dict['positions'].shape[0]))
            elif sort_mode == 'energy':
                particle_indices = np.argsort(particle_dict['energies'])
            else:
                particle_indices = np.arange(particle_dict['positions'].shape[0])  # Cycle through particles for 'normal' mode

        positions_over_time[t] = particle_dict['positions'].copy()
        p_index = particle_indices[mc_index]
        particle_selection = particle_dict_element(particle_dict, p_index)
        total_energy_old = np.sum(particle_dict['energies'])

        # Generate a new position for the particle
        new_pos_proposal = movement_func(particle_selection, radius, movement_scaler, move_mode)
        total_energy_new = np.sum(particle_dict['energies'])

        # Probabilistic acceptance of new position
        k = 1  # Boltzmann constant (normalized)
        alpha = np.min([np.exp(-(total_energy_new - total_energy_old) / (T * k)), 1])
        if (total_energy_new < total_energy_old) or (np.random.uniform() <= alpha):
            update_particle_dict(particle_dict, new_pos_proposal, p_index)

        total_energy_over_time[t] = np.sum(particle_dict['energies'])
        T = cooling_function(T, t, a, b)
        print('timestep (t):', t)

        mc_index += 1

    positions_over_time[-1] = particle_dict['positions'].copy()

    return positions_over_time, total_energy_over_time

# ===== Simulation function ======
def run_simulations(particle_dict, n_sims, param_dict, SA_function):
    """
    Run multiple simulations of particle annealing with varying parameters.

    Args:
    - particles_matrix: The matrix of particles to be used in the simulations.
    - n_sims (int): Number of simulations to run for each parameter combination.
    - param_dict (dict): Dictionary containing different sets of parameters for each simulation.
        Each key in the dictionary represents a parameter name, and its value is a list of values for that parameter.
    - SA_function (function): The function to be used for simulated annealing.

    Returns:
    - results_df (pandas DataFrame): DataFrame containing results of all simulations.
    """


    # Extract parameters for combinations
    sort_mode_list = param_dict.pop('sort_mode_list', ['normal'])
    move_mode_list = param_dict.pop('move_mode_list', ['random cartesian'])
    cooling_functions = param_dict.pop('cooling_function_list', [exponential_decay_cooling])

    # Prepare to collect results
    results = []

    possible_param_combinations = list(product(sort_mode_list, move_mode_list, cooling_functions))
    number_of_possible_param_combination = len(possible_param_combinations)

    print('Possible param combinations:', number_of_possible_param_combination)

    sim_id = 0  # Initialize simulation ID

    for sort_mode, move_mode, cooling_function in possible_param_combinations:
        # Run the simulation
        print('Progress %:', sim_id/(number_of_possible_param_combination*n_sims)*100)
        print('sim_id:', sim_id)

        # Update parameters for current combination
        current_params = {key: value for key, value in param_dict.items()}
        current_params['particle_dict'] = particle_dict  # Adding particles_matrix (input data) to the current params
        current_params['sort_mode'] = sort_mode
        current_params['move_mode'] = move_mode
        current_params['cooling_function'] = cooling_function


        # This variable is only made to remove particles from params when printed
        params_without_data = current_params.copy()
        params_without_data.pop('particle_dict', None)

        print('Running simulation with params:', params_without_data)

        for sim in range(n_sims):
            
            final_positions, total_energy_over_time = SA_function(**current_params)

            # Save the results
            result = {
                'sim_id': sim_id,
                'mode': sort_mode,
                'move_mode': move_mode,
                'cooling_function': cooling_function.__name__ if cooling_function else 'None',
                'final_positions': final_positions,
                'total_energy_over_time': total_energy_over_time
            }
            results.append(result)

            sim_id += 1  # Increment simulation ID for the next run

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df