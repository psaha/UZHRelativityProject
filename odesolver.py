# Plots classical and relativistic orbits for varying initial conditions

import numpy as np
import scipy.integrate
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt


# time delay of light path going past a ... the lensing time delay is always of of order Gm/x**3 (time dimensions) so basically it's proportional to mass to an order magnitude. Sun has Schw radius of 3km and which is 10^-5 light seconds of the order 10 micro seconds so it's plausible that the relativistic signal will be of the order of the schwarzschild radius 
# see how schw radius affects extent of the curves


class Settings(object):

    """Set up parameters describing system."""

    def __init__(self):
        # Set up initial values: [x, y, px, py], number of orbits tested.
        self.initial_values = [20.0, 0.0, 0.0, 2.0] #double mass, major axis and time counts - get same (?check in detail) #0.01 #changed size of orbit from 2000 0 0 0.01 to make harm derivs work - to be investigated further
        self.init_variations = 5
        self.cmpts = 3
        self.total_time = 2*np.pi #450000 #250000
        self.timesteps = 1000
        self.angles = [(n+1)*np.pi/100 for n in range(4)]
        self.number_of_angles = len(self.angles)+1
        self.number_of_curves = self.init_variations**2*self.number_of_angles

    def get_deviations(self):
        """Define deviations from the reference orbit."""
        deviations = np.zeros((self.init_variations, self.init_variations, 4))
        for i in range(self.init_variations):
            for j in range(self.init_variations):
                deviations[i][j] = [0.3*(i-(self.init_variations-1)/2), 0, 0, 0.0005*(j-(self.init_variations-1)/2)]
        return deviations

class InitialConditions(object):

    """Set the initial position and velocity of the reference orbit."""

    def __init__(self, initial_values):
        self.x0 = initial_values[0]
        self.y0 = initial_values[1]
        self.px0 = initial_values[2]
        self.py0 = initial_values[3]

    def initialise_orbit(self, adjustments):
        """Set the initial conditions for orbits from the reference orbit conditions."""
        return [self.x0 + adjustments[0], self.y0 + adjustments[2],
                self.px0 + adjustments[1], self.py0 + adjustments[3]]

def relativistic_derivatives(eqns, tau):
    """
    Relativistic orbit differential equations.

    Returns the differential equations describing movement
    in Schwarzchild spacetime.
    """
    return perturbed_harmonic_derivatives(eqns, tau)

    M = 1
    x = eqns[0]
    y = eqns[1]
    px = eqns[2]
    py = eqns[3]
    r = np.sqrt(x**2 + y**2)
    dx = px - (x*px + y*py)*2*M*x/r**3
    dpx = ((x*px + y*py)*2*M*px/r**3
                - ((1 - 2/r)**-2)*M*x/r**3
                - ((x*px + y*py)**2)*3*M*x/r**5)
    dy = py - (x*px + y*py)*2*M*y/r**3
    dpy = ((x*px + y*py)*2*M*py/r**3
                - ((1 - 2/r)**-2)*M*y/r**3
                - ((x*px + y*py)**2)*3*M*y/r**5)
    return [dx, dy, dpx, dpy]

def classical_derivatives(eqns, tau):

    return perturbed_harmonic_derivatives(eqns, tau, 0)
    """
    Classical orbit differential equations.

    Returns the differential equations describing movement
    under Newtonian gravity.
    """
    M = 1
    x = eqns[0]
    y = eqns[1]
    px = eqns[2]
    py = eqns[3]
    r = np.sqrt(x**2 + y**2)
    dx = px
    dpx = -M*x/r**3
    dy = py
    dpy = -M*y/r**3
    return [dx, dy, dpx, dpy]


def perturbed_harmonic_derivatives(eqns, tau, k=4):
    """
    Classical orbit differential equations.

    Returns the differential equations describing movement
    under Newtonian gravity.
    """
    x = eqns[0]
    y = eqns[1]
    px = eqns[2]
    py = eqns[3]
    r = np.sqrt(x**2 + y**2)
    dx = px
    dpx = -x*(1+((2*k**2)/r**4))
    dy = py
    dpy = -y*(1+((2*k**2)/r**4))
    return [dx, dy, dpx, dpy]

class Orbit_Solution(object):

    """
    Generate orbits and difference orbits.

    Integrates relativistic and classical differential equations
    to obtain orbits for a range of initial conditions and calculates
    the differences between these orbits and a reference orbit.
    """

    def __init__(self, settings):
        self.settings = settings
        self.tau = np.linspace(0.0, self.settings.total_time, self.settings.timesteps)
        self.init = InitialConditions(self.settings.initial_values)
        #self.init_variations = self.settings.init_variations
        #self.number_of_angles = self.settings.number_of_angles

    def solve(self, derivatives, adjustments):
        initial_values = self.init.initialise_orbit(adjustments)
        solution = scipy.integrate.odeint(
                   derivatives, initial_values, self.tau)
        x = solution[:,0]
        y = solution[:,1]
        px = solution[:,2]
        py = solution[:,3]

        return x, y, px, py

    def get_orbits(self):
        deviations = self.settings.get_deviations()
        reference_orbit = np.array(self.solve(classical_derivatives, [0, 0, 0, 0]))
        reference_orbit = reference_orbit[0] + 1j*reference_orbit[1]

        rel_orbits = []
        clas_orbits = []

        for i in range(self.settings.init_variations):
            for j in range(self.settings.init_variations):
                r_orb = self.solve(relativistic_derivatives, deviations[i][j])
                c_orb = self.solve(classical_derivatives, deviations[i][j])

                rel_orbits.append(r_orb)
                clas_orbits.append(c_orb)

        rel_orbits = np.array(rel_orbits)[:,0] + 1j*np.array(rel_orbits)[:,1]
        clas_orbits = np.array(clas_orbits)[:,0] + 1j*np.array(clas_orbits)[:,1]

        return reference_orbit, rel_orbits, clas_orbits

    def get_difference_vectors(self, reference_orbit, rel_orbits, clas_orbits):
        number_of_curves = self.settings.init_variations**2*self.settings.number_of_angles
        rel_differences = np.zeros_like(rel_orbits)
        classical_differences = np.zeros_like(clas_orbits)

        for i in range(self.settings.number_of_curves):
            rel_differences[i] = rel_orbits[i] - reference_orbit
            classical_differences[i] = clas_orbits[i] - reference_orbit
        combined_differences = np.concatenate((rel_differences, classical_differences), axis=0)

        return rel_differences, classical_differences, combined_differences

def rotate_orbit(orbit, theta):
    rotate = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    orbit = np.matrix([orbit.real, orbit.imag])
    rotated_orbit = np.zeros_like(orbit)
    for i in range(orbit.shape[1]):
        rotated_orbit[:,i] = rotate*orbit[:,i]
    rotated_orbit = rotated_orbit[0] + 1j*rotated_orbit[1]
    return rotated_orbit

def get_rotated_orbits(rel_orbits, clas_orbits, settings):
    for i in range(len(rel_orbits)):
        for j in range(len(settings.angles)):
            new_rel = (rotate_orbit(rel_orbits[i], settings.angles[j]))
            new_rel = np.array(new_rel)
            rel_orbits = np.concatenate((rel_orbits, new_rel))
    for i in range(len(clas_orbits)):
        for j in range(len(settings.angles)):
            new_clas = (rotate_orbit(clas_orbits[i], settings.angles[j])) # there was an erroooor here
            new_clas = np.array(new_clas)
            clas_orbits = np.concatenate((clas_orbits, new_clas))

    return rel_orbits, clas_orbits

def basisfuns(A):
    T = A.shape[0]
    lam = np.zeros(shape=T,dtype=float)
    M = A*A.conj().T #A * A+
    U = np.linalg.eigh(M)[1] #eigenvec of M
    sigVT = U.conj().T*A # U+ * A
    VT = 0*sigVT
    for t in range(T):
        f = sigVT[t]*sigVT[t].conj().T
        norm = abs(f[0,0])**0.5
        VT[-1-t] = sigVT[t] / norm
        lam[-1-t] = norm
#    plt.plot(lam)
#    plt.show()
    return lam,VT

def inner_product(matrix1, matrix2):
    c = np.zeros((matrix1.shape[0], matrix2.shape[0]), dtype=complex)
    for n in range(c.shape[0]):
        for m in range(c.shape[1]):
            c[n][m:m+1] = np.dot(matrix1[n], matrix2[m].conj().T)
    return c

def relativistic_components(phi_c, phi_nr):
    #c = inner_product(phi_c, phi_nr)
    delta = 1*phi_c
    for n in range(len(delta)):
        #delta[n] = phi_c[n] - np.dot(c[n], phi_nr) #
        for m in range(len(phi_nr)):
            delta[n] -= np.dot(delta[n],phi_nr[m].conj().T) * phi_nr[m]
    return delta

def plot_orbits(reference_orbit, rel_orbits, clas_orbits):
    # Set up plotting.
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1) #, adjustable='box', aspect = 1.0)
    ax2 = fig.add_subplot(1,2,2) #, adjustable='box', aspect = 1.0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Plot data.
    rel_differences, classical_differences, combined_differences = orbits.get_difference_vectors(
                                                        reference_orbit, rel_orbits, clas_orbits)
    for i in range(len(rel_orbits)):
        ax1.plot(rel_orbits[i].real, rel_orbits[i].imag, 'b')
        ax1.plot(clas_orbits[i].real, clas_orbits[i].imag, 'g')

        ax2.plot(combined_differences[i].real, rel_differences[i].imag, label=('R'))
        ax2.plot(classical_differences[i].real, classical_differences[i].imag, label=('C'))

    ax2.legend()
    plt.show()

def plot_basis_fns(phi_c, phi_nr, str_c, str_nr):
    phi_c_x = phi_c.real
    phi_c_y = phi_c.imag
    phi_nr_x = phi_nr.real
    phi_nr_y = phi_nr.imag
    fig, ax = plt.subplots(nrows=1, ncols=5)
    fig2, ax2 = plt.subplots(nrows=1, ncols=5)
    for i in range(5):
        ax[i].scatter(phi_c_x[i,:], phi_c_y[i,:])
    for i in range(5):
        ax2[i].scatter(phi_nr_x[i,:], phi_nr_y[i,:])

    fig.suptitle('Basis functions for relativistic and classical orbits combined')
    fig2.suptitle('Classical basis functions')
    plt.show()

    plt.plot(str_c)
    plt.plot(str_nr)
    plt.show()

def generate_relativistic_basis(reference_orbit, rel_orbits, clas_orbits, settings):
    rel_differences, classical_differences, combined_differences = orbits.get_difference_vectors(
                                                        reference_orbit, rel_orbits, clas_orbits)
    # Generate differential orbit basis functions (phi_c, phi_nr)
    str_c, phi_c = basisfuns(np.matrix(combined_differences))
    str_nr, phi_nr = basisfuns(np.matrix(classical_differences))

    # Extract relativistic components of the differential orbit basis functions (psi)
    psi = relativistic_components(np.matrix(phi_c[:settings.cmpts]), np.matrix(phi_nr[:settings.cmpts])) #sliced at point where goes to 0

    # Generate basis functions of the relativistic components
    # (should be orthonormal to the non-relativistic components) (psi_basis)
    strength, psi_basis = basisfuns(psi)

    #print(inner_product(psi_basis, phi_nr)) # = zeros! yay!

    # Recreate original orbits from the relativistic components (z from psi)
    # This is a projection of the orbits onto the psi basis
    # Applying basis reconstruction to the classical differences using psi
    # results only in the reference orbit because there are no relativistic components
    # whereas applying to relativistic difference orbits produces different curves
    basis_reconstruction = np.zeros((settings.number_of_curves, settings.timesteps), dtype=complex) #500
    for i in range(settings.number_of_curves):
        rel_dif = np.matrix(rel_differences[i])
        basis_reconstruction[i] = sum([(inner_product(rel_dif, psi_basis[n])*psi_basis[n]) for n in range(settings.cmpts)]) + reference_orbit #vary basis fns to see when get noise fitting 

    return basis_reconstruction



def timeslice(orbit, t):
    new = []
    for i in range(5):
        foo = orbit[i*t:-(5-i)*t]
        new.append(foo)
    return new

################################
############ METHOD ############
################################

# Create orbits (z), differential orbits (z - z_ref)
settings = Settings()
orbits = Orbit_Solution(settings)
reference_orbit, rel_orbits, clas_orbits = orbits.get_orbits()


# TIMESLICER. ISSUE WITH FACT THAT IT'S CUT OFF.
#rel_orbits = timeslice(rel_orbits[0], 100) #You're only timeslicing the first orbit!!!!!!
#clas_orbits = timeslice(clas_orbits[0], 100)
#reference_orbit = reference_orbit[:500] #think about this

# Obtain set of rotated orbits
rel_orbits, clas_orbits = get_rotated_orbits(rel_orbits, clas_orbits, settings)
plot_orbits(reference_orbit, rel_orbits, clas_orbits)

# Reconstruct components of orbits that are purely relativistic and not found in the classical orbits
basis_reconstruction = generate_relativistic_basis(reference_orbit, rel_orbits, clas_orbits, settings)
# Plot reconstructed components
for i in range(len(basis_reconstruction)):
    plt.plot(basis_reconstruction[i].real, basis_reconstruction[i].imag, label=i)
#plt.legend()
plt.show()





#reconstruct rel orb that isnt' part of basis set???







#new_x = reference_orbit[0] + np.array(str_c[0]*phi_c[0].real)
#new_y = reference_orbit[1] + np.array(str_c[0]*phi_c[0].imag)
#new_x2 = reference_orbit[0] + np.array(str_nr[0]*phi_nr[0].real)
#new_y2 = reference_orbit[1] + np.array(str_nr[0]*phi_nr[0].imag)





