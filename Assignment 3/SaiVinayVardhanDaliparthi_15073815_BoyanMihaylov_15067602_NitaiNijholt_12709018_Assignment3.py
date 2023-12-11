import numpy as np
from itertools import combinations


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
        net_forces[p2[0]] += force_vec
    
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
    forces, energies = get_energy_forces_total(positions)

    # Combine positions, forces, and energies into a single dictionary
    particle_dict = {'positions': positions, 'forces': forces, 'energies': energies}

    return particle_dict


def move_particle_radial(particle_dict, radius=1, movement_scaler=1, move_mode = 'random absolute'):
    """Takes a particle which is a positional tuple of (x,y) and returns the new position of this particle
    """

    if move_mode == 'random absolute':

        # Determine target position by randomly sampling polar coordinates within global circle
        theta_new = np.random.uniform(0, 2*np.pi)
        r_new = np.random.uniform(0, radius)

        x_new = r_new*np.cos(theta_new)
        y_new =  r_new*np.sin(theta_new)

        pos_new = np.array([x_new, y_new])

        # Scale perturbation vector by movement scaler
        delta_vec = (pos_new - particle_dict['positions']) * movement_scaler

        particle_dict['positions'] += delta_vec
    
    elif move_mode == 'random relative':
        
        # Create random polar perturbation vector around current position
        delta_theta = np.random.uniform(0, 2*np.pi)
        delta_r = np.random.uniform(0, radius * 2)
        
        # Construct perturbation vector and scale by movement scaler
        delta_x = delta_r*np.cos(delta_theta)
        delta_y =  delta_r*np.sin(delta_theta)
        delta_vec = np.array([delta_x, delta_y]) * movement_scaler

        # Limit to circle boundary

        particle_dict['positions'] = np.array([x_new, y_new])

    elif move_mode == 'repell':
        
        # Determine perturbation vector from 
        delta_vec = particle_dict['forces']
