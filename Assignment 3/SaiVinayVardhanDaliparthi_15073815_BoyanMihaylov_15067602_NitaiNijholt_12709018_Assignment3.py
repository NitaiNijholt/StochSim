import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
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


def move_particle_radial(particle_dict, radius=1, movement_scaler=1, move_mode = 'random absolute', step_scale=None):
    """Takes an array of particle dictionaries and displaces the
    particle positions along the particle velocity vectors.
    Updates forces and energies once the displacement is done
    """

    if move_mode == 'random absolute':

        # Determine target position by randomly sampling polar coordinates within global circle
        theta_new = np.random.uniform(0, 2*np.pi, particle_dict['positions'].shape[0])
        r_new = np.random.uniform(0, radius, particle_dict['positions'].shape[0])

        x_new = r_new*np.cos(theta_new)
        y_new =  r_new*np.sin(theta_new)

        pos_new = np.vstack((x_new, y_new)).T

        # Scale perturbation vector by movement scaler
        delta_vec = (pos_new - particle_dict['positions']) * movement_scaler

        particle_dict['positions'] += delta_vec
    
    elif move_mode == 'random relative':
        
        # Create random polar perturbation vector around current position
        delta_theta = np.random.uniform(0, 2*np.pi, particle_dict['positions'].shape[0])
        delta_r = np.random.uniform(0, radius * 2, particle_dict['positions'].shape[0])
        
        # Construct perturbation vector and scale by movement scaler
        delta_x = delta_r*np.cos(delta_theta)
        delta_y =  delta_r*np.sin(delta_theta)
        delta_vec = np.vstack((delta_x, delta_y)).T * movement_scaler

        # particle_dict['positions'] += delta_vec
        particle_dict['positions'] = reflect_at_circle_bounds(particle_dict['positions'], particle_dict['forces'] * movement_scaler, radius)

    elif move_mode == 'repell':

        particle_dict['positions'] = reflect_at_circle_bounds(particle_dict['positions'], particle_dict['forces'] * movement_scaler, radius)

    # Update energies / forces
    energies, forces = get_energy_forces_total(particle_dict['positions'])
    particle_dict['energies'] = energies
    particle_dict['forces'] = forces


def visualise_particles(particle_dict, radius=1, movement_scaler=1, title="Particle positions and velocities", ax=None):
    """Creates a plot visualising the particle positions and velocities
    """

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Visualise particle positions
    coords = particle_dict['positions'].T
    artist = ax.scatter(coords[0], coords[1], s=10, c=particle_dict['energies'])

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
    ax.set_xlim((-1.25, 1.25))
    ax.set_ylim((-1.25, 1.25))

    # Setting labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    fig.suptitle(title)

    if not ax:
        # Show the plot
        plt.show()