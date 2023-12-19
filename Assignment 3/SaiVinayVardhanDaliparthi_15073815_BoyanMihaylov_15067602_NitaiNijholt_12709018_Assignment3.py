import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import animation as anim
import seaborn as sns
from itertools import combinations, product
import pandas as pd


# ===== Particle operation functions =====
def get_energy_force_2_particles(p_i, p_j, coulomb_constant=1, energy_only=False):
    """
    Calculate the electrostatic force between two particles and
    calculate the potential energy between two particles based on Coulomb's Law.

    Args:
        p_i (tuple): Position (x, y) of the first particle.
        p_j (tuple): Position (x, y) of the second particle.
        coulomb_constant (float, optional): Coulomb's constant, default is 1 for normalized calculations.
        energy_only (bool, optional): Whether to return only the energy or also force. Defaults to False.

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
    if energy_only:
        force_i_j = None
    else:
        force_i_j = coulomb_constant * r_i_j / (dist_i_j**3)

    # Calculate the potential energy using Coulomb's Law
    energy_i_j = coulomb_constant / dist_i_j

    return energy_i_j, force_i_j


def get_energy_forces_total(particles, energy_only=False):
    """
    Calculate the total potential energy of a system of particles.
    Sum all the force vectors acting on each particle
    
    Args:
        particles (list or numpy.ndarray): A list of tuples or an n x 2 matrix representing the positions of n particles.
        energy_only (bool, optional): Whether to return only the energy or also the net force on each particle. Defaults to False.

    Returns:
        float: The total potential energy of the system.
    """

    # Convert list of tuples to numpy array if it isn't already
    if not isinstance(particles, np.ndarray):
        particles = np.array([list(row) for row in particles])

    # Initialize array to store energies for each particle
    energies = np.zeros(len(particles))

    # Initialize an array to store the net force on each particle
    if energy_only:
        net_forces = None
    else:
        net_forces = np.zeros_like(particles)  # Assuming 2D particles

    # Calculate energy between each unique pair of particles
    for p1, p2 in combinations(enumerate(particles), 2):
        energy_pair, force_vec = get_energy_force_2_particles(p1[1], p2[1])
        energies[p1[0]] += energy_pair*0.5
        energies[p2[0]] += energy_pair*0.5
        if not energy_only:
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


def update_particle_dict(particle_dict, positions, indices):
    """Updates the particle dictionary based new positions at specific indices

    Args:
        particle_dict (dict): An object containing the positions, forces and energies of all particles
        positions (numpy.ndarray): An array of new positions
        indices (list): A list of indices to update
    """
    
    particle_dict['positions'][indices] = positions
    # for i, index in enumerate(indices):
    #     particle_dict['positions'][index] = positions[i]
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


def move_particle(particle_dict, radius=1, movement_scaler=1, blend_p=0.5, move_mode = 'repell'):
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

            # Construct and normalise perturbation vector
            delta_vec = pos_new - particle_dict['positions'][i]
            # delta_vec /= np.linalg.norm(pos_new - particle_dict['positions'][i])

            # Scale perturbation vector by movement scaler
            # delta_vec *= movement_scaler

            new_positions[i] += delta_vec

    elif move_mode == 'random polar absolute':  # Random polar coordinates within circle

        # Determine target position by randomly sampling polar coordinates within global circle
        theta_new = np.random.uniform(0, 2*np.pi, particle_dict['positions'].shape[0])
        r_new = np.random.uniform(0, radius, particle_dict['positions'].shape[0])

        x_new = r_new*np.cos(theta_new)
        y_new =  r_new*np.sin(theta_new)

        pos_new = np.vstack((x_new, y_new)).T

        # Construct and normalise perturbation vector
        delta_vec = pos_new - particle_dict['positions']
        # delta_vec /= np.linalg.norm(pos_new - particle_dict['positions'])

        # Scale perturbation vector by movement scaler
        # delta_vec *= movement_scaler

        new_positions += delta_vec
    
    elif move_mode == 'random polar relative':  # Random polar coordinates relative to current position
        
        # Create random polar perturbation vector around current position
        delta_theta = np.random.uniform(0, 2*np.pi, particle_dict['positions'].shape[0])
        delta_r = np.random.uniform(0, radius, particle_dict['positions'].shape[0])
        
        # Construct perturbation vector
        delta_x = delta_r*np.cos(delta_theta)
        delta_y =  delta_r*np.sin(delta_theta)
        delta_vec = np.vstack((delta_x, delta_y)).T

        # particle_dict['positions'] += delta_vec
        new_positions = reflect_at_circle_bounds(particle_dict['positions'], delta_vec * movement_scaler, radius)

    elif move_mode == 'repell':

        # Normalise force vectors
        force_vecs = particle_dict['forces'] / np.linalg.norm(particle_dict['forces'], axis=1)[:, np.newaxis]

        new_positions = reflect_at_circle_bounds(particle_dict['positions'], force_vecs * movement_scaler, radius)

    elif move_mode == 'blend':

        # Randomly select between random absolute and repell
        new_pos_det = move_particle(particle_dict, radius, movement_scaler, move_mode='repell')
        new_pos_rand = move_particle(particle_dict, radius, movement_scaler, move_mode='random polar absolute')
        dice_roll = np.random.uniform(size=particle_dict['positions'].shape[0])
        dice_roll = np.reshape(dice_roll, (dice_roll.shape[0], 1))
        new_positions = np.where(dice_roll < blend_p, new_pos_rand, new_pos_det)

    else:
        raise ValueError("move_mode must be one of 'random cartesian', 'random polar absolute', 'random polar relative', 'repell'")

    return new_positions


def particle_dict_element(particle_dict, i):
    """Returns a dictionary containing the position, energy and force of a single particle
    """
    selection_dict = {}
    for keys in particle_dict.keys():
        selection_dict[keys] = np.array([particle_dict[keys][i]])
    return selection_dict


# ===== Visualisation functions =====
def visualise_particles(particle_dict, radius=1, vector_scale=1, title="Particle positions and velocities", ax=None):
    """Creates a plot visualising the particle positions and velocities
    """

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Re-convert to numpy arrays
    particle_dict['positions'] = np.array(particle_dict['positions'])
    particle_dict['forces'] = np.array(particle_dict['forces'])
    particle_dict['energies'] = np.array(particle_dict['energies'])
    
    # Visualise particle forces
    for i in range(particle_dict['forces'].shape[0]):
        pos = particle_dict['positions'][i]
        dir = vector_scale * particle_dict['forces'][i]
        ax.arrow(pos[0], pos[1], dir[0], dir[1], width=0.02, head_width=0.2, color='lightgrey')
        ax.annotate(i, pos)

    # Visualise particle positions
    coords = particle_dict['positions'].T
    ax.scatter(coords[0], coords[1], s=40, c=particle_dict['energies'])

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


def visualise_particles_at_sim_times(df_sim_results, local_sim_id, times, radius=1, vector_scale=0.1):
    """Creates a plot visualising the particle positions and velocities at specified times
    """

    # Create separate DataFrame rows for each simulation step
    df_sim_results_exploded = unpack_time_data(df_sim_results)

    # Construct particle selection dictionary for specific simulations and times
    particle_selection = df_sim_results_exploded[(df_sim_results_exploded['local_sim_id'] == local_sim_id) & (df_sim_results_exploded['time'].isin(times))]
    
    # Set up plots for each parameter combination
    sort_modes = df_sim_results_exploded['sort_mode'].unique()
    move_modes = df_sim_results_exploded['move_mode'].unique()
    cooling_functions = df_sim_results_exploded['cooling_function'].unique()
    markov_lengths = df_sim_results_exploded['markov_length'].unique()
    possible_param_combinations = list(product(sort_modes, move_modes, cooling_functions, markov_lengths))
    fig, axs = plt.subplots(len(possible_param_combinations), len(times))
    fig.set_size_inches(3*len(times), 3.5*len(possible_param_combinations))

    # Plot particles for each parameter combination
    for i, (sort_mode, move_mode, cooling_function, markov_length) in enumerate(possible_param_combinations):
        for j, time in enumerate(times):
            ax = axs[i, j]
            ax.set_title(f"t={time}\nS: {sort_mode}\nM: {move_mode}\nC: {cooling_function}\nL: {markov_length}")
            particle_param_combination = particle_selection[(particle_selection['sort_mode'] == sort_mode) &
                                                            (particle_selection['move_mode'] == move_mode) &
                                                            (particle_selection['cooling_function'] == cooling_function) &
                                                            (particle_selection['markov_length'] == markov_length)]
            positions = particle_param_combination['positions_over_time'].values[0][time]
            forces = particle_param_combination['forces_over_time'].values[0][time]
            energies = particle_param_combination['energies_over_time'].values[0][time]
            particle_dict = {'positions': positions, 'forces': forces, 'energies': energies}
            # print(particle_dict['forces'])
            visualise_particles(particle_dict, radius=radius, vector_scale=vector_scale, ax=ax)
    plt.tight_layout(pad=3.0)
    plt.show()


def plot_energy_landscape(positions, radius=1.0, ax=None):
    """Plot a 3D surface representation of the energy field of the particle system
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Recalculate energies at positions
    energies, _ = get_energy_forces_total(positions, energy_only=True)

    # Create a meshgrid of x and y values
    x = np.linspace(-radius, radius, 100)
    y = np.linspace(-radius, radius, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate potential energy for each point in the meshgrid
    # based on distances to input positions
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            grid_pt = np.array([X[i, j], Y[i, j]])
            closest_pos_i = np.argmin(np.linalg.norm(positions - grid_pt, axis=1))
            energy_total_pt = 0
            for k, pos in enumerate(positions):
                if k != closest_pos_i:
                    energy_total_pt += 0.5* get_energy_force_2_particles(grid_pt, pos, energy_only=True)[0]
            Z[i, j] = energy_total_pt
    
    # Create a circle in the xy plane at z=0
    theta = np.linspace(0, 2*np.pi, 100)
    r = radius  # radius of the circle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(theta)  # z=0 for all points on the circle

    # Plot the circle
    ax.plot(x, y, z, color='grey')

    # Plot the energy field
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.75)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Energy')

    # Plot particle positions and their projections
    ax.scatter(positions[:, 0], positions[:, 1], energies, color='black')
    ax.scatter(positions[:, 0], positions[:, 1], np.zeros_like(energies), color='grey', marker='x')
    for i, pos in enumerate(positions):
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, energies[i]], color='grey', linestyle='--')

    ax.set_zlim((0,10))

    if not ax:
        plt.show()


def plot_convergence(df_sim_results):
    """Plots a comparison of the total energy convergence for different simulation strategies

    Args:
        df_sim_results (pandas.DataFrame): A DataFrame containing the results of the simulations
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
    """

    fig, axs = plt.subplots(df_sim_results['cooling_function'].unique().shape[0])
    fig.set_size_inches(10, 10)

    # Create separate DataFrame rows for each simulation step
    df_sim_results_exploded = unpack_time_data(df_sim_results)

    for i, cooling_function in enumerate(df_sim_results['cooling_function'].unique()):
        axs[i].set_title(f"Energy convergence for {cooling_function} cooling")
        df_cooling = df_sim_results_exploded[df_sim_results['cooling_function'] == cooling_function]
        sns.lineplot(data=df_cooling, x=df_cooling['time'], y=df_cooling['total_energy'],
                     hue=df_cooling['sort_mode'], style=df_cooling['move_mode'], ax=axs[i])
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Total energy')
    
    plt.tight_layout()
    plt.show()


# ===== Cooling schedule functions =====
def logarithmic_decay_cooling(T_init, t, a, b, cr, std):
    """https://canvas.uva.nl/courses/39303/pages/lecture-09-the-gibbs-sampler-and-simulated-annealing?module_item_id=1830823 slide 14"""
    T_n = a/(np.log(t+b))
    return T_n

def parametric_exponential_decay_cooling(T_init, t, a, b, cr, std):
    """Cooling rate is a function of time
    """
    return T_init * (a ** t)

def exponential_decay_cooling(T_init, t, a, b, cr, std):
    """cr is the cooling rate
    """
    return T_init * cr

def std_cooling(T_init, t, a, b, cr, std):
    """cr = (1 - delta), delta controlling how much the probability functions
    should differ between two chains.
    std is the standard deviation of the energy"""
    return T_init / (1 + (T_init * np.log(2 - cr) / (3 * std)))

# ====== Simulated annealing functions ======
def sim_annealing_move_particles(particle_dict, time_range, markov_length, radius, T_init=1, cooling_function=exponential_decay_cooling, a=100, b=1, cr=0.995, movement_func=move_particle, sort_mode='normal', move_mode='random', movement_scaler=1, decrease_step=False):
    """
    Perform simulated annealing to update particle positions within a specified radius.
    Per each time iteration a Markov chain is iterated through multiple times.

    Args:
        particle_dict (dict): An object containing the positions, forces and energies of all particles
        time_range (int): Number of time steps to run the simulation for
        markov_length (int): Length of Markov chain to run per time step
        radius (float): The radius of the circle
        T_init (float, optional): Initial temperature. Defaults to 1.
        cooling_function (function, optional): Cooling function to use. Defaults to exponential_decay_cooling.
        a (float, optional): Parameter for cooling function. Defaults to 100.
        b (float, optional): Parameter for cooling function. Defaults to 1.
        cr (float, optional): Parameter for cooling function. Defaults to 0.995.
        movement_func (function, optional): Function to use for moving particles. Defaults to move_particle.
        sort_mode (str, optional): Mode to use for sorting particles. Defaults to 'normal'.
        move_mode (str, optional): Mode to use for moving particles. Defaults to 'random'.
        movement_scaler (float, optional): Scaling factor for movement. Defaults to 1.
        decrease_step (bool, optional): Whether to decrease the movement scaler over time. Defaults to False.
    """

    # Create running copy of particle_dict
    particle_dict = particle_dict.copy()

    if cooling_function is None:
        raise ValueError("cooling_function must be provided")

    T = T_init
    total_energy_over_time = np.zeros(time_range)
    positions_over_time = np.zeros((time_range + 1, particle_dict['positions'].shape[0], 2))
    forces_over_time = np.zeros((time_range + 1, particle_dict['forces'].shape[0], 2))
    energies_over_time = np.zeros((time_range + 1, particle_dict['energies'].shape[0]))

    for t in range(time_range):
        
        # Optionally decrease the movement scaler over time
        if decrease_step:
            movement_scaler = movement_scaler * (1 - 2**(t*10/time_range - 10))
        
        # At the beginning of each Markov chain, sort indices of particles to move based on the mode
        if sort_mode == 'random':
            particle_indices = np.random.permutation(np.arange(particle_dict['positions'].shape[0])) # Randomly shuffle particles
        elif sort_mode == 'energy':
            particle_indices = np.flip(np.argsort(particle_dict['energies']))  # Sort particles by energy (descending)
        else:
            particle_indices = np.arange(particle_dict['positions'].shape[0])  # Cycle through particles for 'normal' mode

        # Store current positions, forces and energies
        positions_over_time[t] = particle_dict['positions'].copy()
        forces_over_time[t] = particle_dict['forces'].copy()
        energies_over_time[t] = particle_dict['energies'].copy()
        
        total_energy_chain = np.zeros(markov_length)
        for m in range(markov_length):
            
            # Cycle through particle indices
            p_index = particle_indices[m % particle_dict['positions'].shape[0]]
            particle_selection = particle_dict_element(particle_dict, p_index)
            total_energy_old = np.sum(particle_dict['energies'])

            # Generate a new position for the particle
            new_pos_proposal = movement_func(particle_selection, radius, movement_scaler, T/T_init, move_mode)
            new_dict_proposal = particle_dict.copy()
            update_particle_dict(new_dict_proposal, new_pos_proposal, [p_index])
            total_energy_new = np.sum(new_dict_proposal['energies'])

            # Probabilistic acceptance of new position
            k = 1  # Boltzmann constant (normalized)
            alpha = np.min([np.exp(-(total_energy_new - total_energy_old) / (T * k)), 1])
            if (total_energy_new < total_energy_old) or (np.random.uniform() <= alpha):
                particle_dict = new_dict_proposal.copy()
                total_energy_chain[m] = total_energy_new
            else:
                total_energy_chain[m] = total_energy_old

        # Update temperature
        if total_energy_chain.shape[0] > 3:
            energy_std = np.std(total_energy_chain)
        else:
            energy_std = 1
        T = cooling_function(T, t, a, b, cr, energy_std)
        # print('timestep (t):', t)

        total_energy_over_time[t] = total_energy_chain[-1]

    positions_over_time[-1] = particle_dict['positions'].copy()
    forces_over_time[-1] = particle_dict['forces'].copy()
    energies_over_time[-1] = particle_dict['energies'].copy()

    return positions_over_time, forces_over_time, energies_over_time, total_energy_over_time


def sim_annealing_move_particles_parallel(particle_dict, time_range, markov_length, radius, T_init=1, cooling_function=exponential_decay_cooling, a=100, b=1, cr=0.995, movement_func=move_particle, sort_mode='normal', move_mode='random', movement_scaler=1, decrease_step=False):
    """
    Perform simulated annealing to update particle positions within a specified radius.
    Unlike the previous function, this function perturbs multiple particles at once.
    Per each time iteration a Markov chain is iterated through multiple times.

    Args:
        particle_dict (dict): An object containing the positions, forces and energies of all particles
        time_range (int): Number of time steps to run the simulation for
        markov_length (int): Length of Markov chain to run per time step
        radius (float): The radius of the circle
        T_init (float, optional): Initial temperature. Defaults to 1.
        cooling_function (function, optional): Cooling function to use. Defaults to exponential_decay_cooling.
        a (float, optional): Parameter for cooling function. Defaults to 100.
        b (float, optional): Parameter for cooling function. Defaults to 1.
        cr (float, optional): Parameter for cooling function. Defaults to 0.995.
        movement_func (function, optional): Function to use for moving particles. Defaults to move_particle.
        sort_mode (str, optional): Mode to use for sorting particles. Defaults to 'normal'.
        move_mode (str, optional): Mode to use for moving particles. Defaults to 'random'.
        movement_scaler (float, optional): Scaling factor for movement. Defaults to 1.
        decrease_step (bool, optional): Whether to decrease the movement scaler over time. Defaults to False.
    """

    # Create running copy of particle_dict
    particle_dict = particle_dict.copy()

    if cooling_function is None:
        raise ValueError("cooling_function must be provided")

    T = T_init
    total_energy_over_time = np.zeros(time_range)
    positions_over_time = np.zeros((time_range + 1, particle_dict['positions'].shape[0], 2))
    forces_over_time = np.zeros((time_range + 1, particle_dict['forces'].shape[0], 2))
    energies_over_time = np.zeros((time_range + 1, particle_dict['energies'].shape[0]))

    for t in range(time_range):
        
        # Optionally decrease the movement scaler over time
        if decrease_step:
            movement_scaler = movement_scaler * (1 - 2**(t*10/time_range - 10))

        # Store current positions, forces and energies
        positions_over_time[t] = particle_dict['positions'].copy()
        forces_over_time[t] = particle_dict['forces'].copy()
        energies_over_time[t] = particle_dict['energies'].copy()

        total_energy_chain = np.zeros(markov_length)
        for m in range(markov_length):
            
            # Cycle through particle indices
            # p_index = particle_indices[m % particle_dict['positions'].shape[0]]
            # particle_selection = particle_dict_element(particle_dict, p_index)
            total_energy_old = np.sum(particle_dict['energies'])

            # Generate a new position for the particle
            new_pos_proposal = movement_func(particle_dict, radius, movement_scaler, T/T_init, move_mode)
            energies_new, _ = get_energy_forces_total(new_pos_proposal, True)
            total_energy_new = np.sum(energies_new)

            # Probabilistic acceptance of new position
            k = 1  # Boltzmann constant (normalized)
            alpha = np.min([np.exp(-(total_energy_new - total_energy_old) / (T * k)), 1])
            if (total_energy_new < total_energy_old) or (np.random.uniform() <= alpha):
                update_particle_dict(particle_dict, new_pos_proposal, np.arange(particle_dict['positions'].shape[0]))
                total_energy_chain[m] = total_energy_new
            else:
                total_energy_chain[m] = total_energy_old

        # Update temperature
        if total_energy_chain.shape[0] > 3:
            energy_std = np.std(total_energy_chain)
        else:
            energy_std = 1
        T = cooling_function(T, t, a, b, cr, energy_std)
        # print('timestep (t):', t)

        total_energy_over_time[t] = total_energy_chain[-1]

    positions_over_time[-1] = particle_dict['positions'].copy()
    forces_over_time[-1] = particle_dict['forces'].copy()
    energies_over_time[-1] = particle_dict['energies'].copy()

    return positions_over_time, forces_over_time, energies_over_time, total_energy_over_time


# ===== Simulation function ======
def run_simulations(n_sims, param_dict, SA_function):
    """
    Run multiple simulations of particle annealing with varying parameters.

    Args:
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
    markov_length_list = param_dict.pop('markov_length_list', [1])

    # Extract general parameters
    n_particles = param_dict.pop('n_particles', 10)
    radius = param_dict.pop('radius', 1)
    decrease_step = param_dict.pop('decrease_step', False)

    # Prepare to collect results
    results = []

    possible_param_combinations = list(product(sort_mode_list, move_mode_list, cooling_functions, markov_length_list))
    number_of_possible_param_combination = len(possible_param_combinations)

    print('Possible param combinations:', number_of_possible_param_combination)

    sim_id = 0  # Initialize simulation ID

    for sort_mode, move_mode, cooling_function, markov_length in possible_param_combinations:
        # Run the simulation
        print('Progress %:', sim_id/(number_of_possible_param_combination*n_sims)*100)
        print('sim_id:', sim_id)

        # Update parameters for current combination
        current_params = {key: value for key, value in param_dict.items()}
        current_params['particle_dict'] = initialise_particle_dict_random(n_particles, radius) # Initialise particle positions
        current_params['sort_mode'] = sort_mode
        current_params['move_mode'] = move_mode
        current_params['cooling_function'] = cooling_function
        current_params['markov_length'] = markov_length
        current_params['radius'] = radius
        current_params['decrease_step'] = decrease_step

        # This variable is only made to remove particles from params when printed
        params_without_data = current_params.copy()
        params_without_data.pop('particle_dict', None)

        print('Running simulation with params:', params_without_data)

        for local_sim_id in range(n_sims):
            
            pos_over_time, forces_over_time, energies_over_time, total_energy_over_time = SA_function(**current_params)
            # final_positions = pos_over_time[-1]
            # final_total_evergy = total_energy_over_time[-1]

            # Save the results
            result = {
                'sim_id': sim_id,
                'local_sim_id': local_sim_id,
                'sort_mode': sort_mode,
                'move_mode': move_mode,
                'cooling_function': cooling_function.__name__ if cooling_function else 'None',
                'markov_length': markov_length,
                'positions_over_time': pos_over_time,
                'forces_over_time': forces_over_time,
                'energies_over_time': energies_over_time,
                'total_energy_over_time': total_energy_over_time
            }
            results.append(result)

            sim_id += 1  # Increment simulation ID for the next run

    print("Done.")

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df


def unpack_time_data(run_data):
    """Unpacks the time data from a DataFrame containing simulation results

    Args:
        run_data (pandas.DataFrame): A DataFrame containing simulation results

    Returns:
        pandas.DataFrame: A DataFrame containing the time data for each simulation
    """

    # Create separate DataFrame rows for each simulation step
    run_data['total_energy_timestep'] = run_data['total_energy_over_time'].apply(lambda x: list(enumerate(x)))
    run_data_exploded = run_data.explode('total_energy_timestep')

    # Split the tuple into two separate columns
    run_data_exploded[['time', 'total_energy']] = pd.DataFrame(run_data_exploded['total_energy_timestep'].tolist(), index=run_data_exploded.index)

    return run_data_exploded