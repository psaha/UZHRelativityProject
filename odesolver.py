
import numpy as np
import scipy.integrate
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt



class Settings(object):

    """Set up parameters describing system."""

    def __init__(self):
        """Set up initial values: [x, y, px, py], number of orbits tested."""
        a = 3e3
        e = 0.9
        x_apo = a*(1+e)
        py_apo = np.sqrt((1-e)/x_apo)
        self.initial_values = [x_apo, 0.0, 0.0, py_apo]
        self.init_variations = 5
        self.cmpts = 10
        self.timesteps = 11000 #10330 is the number to get a just closed orbit
        self.total_time = self.timesteps*100
        self.angles = [(n+1)*np.pi/100 for n in range(4)]
        self.number_of_angles = len(self.angles)+1
        self.slices = 5  #NB if I change this to 6 then it breaks, probably due to integer division!
        self.number_of_curves = self.init_variations**2*self.number_of_angles*self.slices
        self.perturber_mass = 0.1

    def get_deviations(self):
        """Define deviations from the reference orbit initial position and momentum."""
        deviations = np.zeros((self.init_variations, self.init_variations, 4))
        for i in range(self.init_variations):
            for j in range(self.init_variations):
                deviations[i][j] = [30*(i-(self.init_variations-1)/2), 0, 0, 0.0001*(j-(self.init_variations-1)/2)]

        return deviations


class InitialConditions(object):

    """Set the initial position and velocity of the reference orbit."""

    def __init__(self, initial_values):
        """Input initial position and momentum for the reference orbit."""
        self.x0 = initial_values[0]
        self.y0 = initial_values[1]
        self.px0 = initial_values[2]
        self.py0 = initial_values[3]

    def initialise_orbit(self, adjustments):
        """Set the initial conditions for the set of orbits from the reference orbit conditions."""
        return [self.x0 + adjustments[0], self.y0 + adjustments[2],
                self.px0 + adjustments[1], self.py0 + adjustments[3]]

def relativistic_derivatives(eqns, tau, m):
    """
    Relativistic orbit differential equations.

    Returns the differential equations describing movement
    in Schwarzchild spacetime. Includes a Newtonian
    perturbing body moving in a circular orbit.
    """
    #return perturbed_harmonic_derivatives(eqns, tau)

    M = 1
    r_n = 100
    x_n = -r_n*np.sin(tau/r_n**1.5) #r_n*np.cos(tau/r_n**1.5)
    y_n = r_n*np.cos(tau/r_n**1.5) #r_n*np.sin(tau/r_n**1.5)
    x = eqns[0]
    y = eqns[1]
    px = eqns[2]
    py = eqns[3]
    r = np.sqrt(x**2 + y**2)
    denom1 = r_n**3
    denom2 = ((x-x_n)**2 + (y-y_n)**2)**1.5
    dx = px - (x*px + y*py)*2*M*x/r**3
    dpx = ((x*px + y*py)*2*M*px/r**3
                - ((1 - 2/r)**-2)*M*x/r**3
                - ((x*px + y*py)**2)*3*M*x/r**5
                - m*(x_n/denom1 + (x-x_n)/denom2))
    dy = py - (x*px + y*py)*2*M*y/r**3
    dpy = ((x*px + y*py)*2*M*py/r**3
                - ((1 - 2/r)**-2)*M*y/r**3
                - ((x*px + y*py)**2)*3*M*y/r**5
                - m*(y_n/denom1 + (y-y_n)/denom2))
    return [dx, dy, dpx, dpy]

def classical_derivatives(eqns, tau, m):

    """
    Classical orbit differential equations.

    Returns the differential equations describing movement
    under Newtonian gravity. Includes a Newtonian
    perturbing body moving in a circular orbit.
    """
    #return perturbed_harmonic_derivatives(eqns, tau, 0)

    M = 1
    r_n = 100
    x_n = -r_n*np.sin(tau/r_n**1.5) #r_n*np.cos(tau/r_n**1.5)
    y_n = r_n*np.cos(tau/r_n**1.5) #r_n*np.sin(tau/r_n**1.5)
    x = eqns[0]
    y = eqns[1]
    px = eqns[2]
    py = eqns[3]
    r = np.sqrt(x**2 + y**2)
    denom1 = r_n**3
    denom2 = ((x-x_n)**2 + (y-y_n)**2)**1.5
    dx = px
    dpx = -M*x/r**3 - m*(x_n/denom1 + (x-x_n)/denom2)
    dy = py
    dpy = -M*y/r**3 - m*(y_n/denom1 + (y-y_n)/denom2)
    return [dx, dy, dpx, dpy]


def perturbed_harmonic_derivatives(eqns, tau, k=4):
    """
    Differential equations for a perturbed harmonic potential.

    Returns the differential equations describing movement
    under a harmonic potential with a variable perturbation applied.
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

    def solve(self, derivatives, adjustments, m, t, start_point):
        """Solve differential equations to generate orbit."""
        initial_values = InitialConditions(start_point).initialise_orbit(adjustments)
        solution = scipy.integrate.odeint(
                   derivatives, initial_values, t, args=(m,))
        x = solution[:,0]
        y = solution[:,1]
        px = solution[:,2]
        py = solution[:,3]

        return x, y, px, py

    def get_reference_orbit(self):
        """Generate reference orbit and the conditions at the start point for the timeslicing."""
        initial_values = InitialConditions(self.settings.initial_values).initialise_orbit([0, 0, 0, 0])
        solution = scipy.integrate.odeint(
                   classical_derivatives, initial_values, self.tau, args=(0,))
        x = solution[:,0]
        y = solution[:,1]
        px = solution[:,2]
        py = solution[:,3]

        reference_orbit = np.array([x, y, px, py])
        start_point = reference_orbit[:,-100]

        reference_orbit = reference_orbit[0] + 1j*reference_orbit[1]

        return reference_orbit, start_point

    def get_orbits(self):
        """Obtain reference orbit and orbit set for varied position, momentum and orientation."""
        time = np.linspace(0.0, self.settings.total_time+20000, self.settings.timesteps+200)
        deviations = self.settings.get_deviations()
        reference_orbit, start_point = self.get_reference_orbit()

        rel_orbits = []
        clas_orbits = []

        for i in range(self.settings.init_variations):
            for j in range(self.settings.init_variations):
                r_orb = self.solve(relativistic_derivatives, deviations[i][j], self.settings.perturber_mass, time, start_point)
                c_orb = self.solve(classical_derivatives, deviations[i][j], self.settings.perturber_mass, time, start_point)

                rel_orbits.append(r_orb)
                clas_orbits.append(c_orb)

        rel_orbits = np.array(rel_orbits)[:,0] + 1j*np.array(rel_orbits)[:,1]
        clas_orbits = np.array(clas_orbits)[:,0] + 1j*np.array(clas_orbits)[:,1]

        if self.settings.angles != []:
            rel_orbits, clas_orbits = self.get_rotated_orbits(rel_orbits, clas_orbits)
        # for plotting purposes
        unpeturbed_rel = orbits.solve(relativistic_derivatives, deviations[0][0], 0, time, start_point)
        unpeturbed_rel = np.array(unpeturbed_rel)[0] + 1j*np.array(unpeturbed_rel)[1]

        return reference_orbit, rel_orbits, clas_orbits, unpeturbed_rel

    def get_difference_vectors(self, reference_orbit, rel_orbits, clas_orbits):
        """Return difference between orbit and the reference orbit."""
        number_of_curves = self.settings.number_of_curves
        rel_differences = np.zeros_like(rel_orbits)
        classical_differences = np.zeros_like(clas_orbits)

        for i in range(self.settings.number_of_curves):
            rel_differences[i] = rel_orbits[i] - reference_orbit
            classical_differences[i] = clas_orbits[i] - reference_orbit
        combined_differences = np.concatenate((rel_differences, classical_differences), axis=0)

        return rel_differences, classical_differences, combined_differences

    def rotate_orbit(self, orbit, theta):
        """Perform a rotation on an orbit by an angle theta."""
        rotate = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        orbit = np.matrix([orbit.real, orbit.imag])
        rotated_orbit = np.zeros_like(orbit)
        for i in range(orbit.shape[1]):
            rotated_orbit[:,i] = rotate*orbit[:,i]
        rotated_orbit = rotated_orbit[0] + 1j*rotated_orbit[1]
        return rotated_orbit

    def get_rotated_orbits(self, rel_orbits, clas_orbits):
        """Take set of orbits and extend it to include rotations of that set."""
        for i in range(len(rel_orbits)):
            for j in range(len(settings.angles)):
                new_rel = (self.rotate_orbit(rel_orbits[i], self.settings.angles[j]))
                new_rel = np.array(new_rel)
                rel_orbits = np.concatenate((rel_orbits, new_rel))
        for i in range(len(clas_orbits)):
            for j in range(len(settings.angles)):
                new_clas = (self.rotate_orbit(clas_orbits[i], self.settings.angles[j]))
                new_clas = np.array(new_clas)
                clas_orbits = np.concatenate((clas_orbits, new_clas))

        return rel_orbits, clas_orbits

def timeslice(orbits):
    """Take slices of the set of orbits from different time starting points."""
    settings = Settings()
    time_length = 200//settings.slices
    orbit_collection = []
    for i in range(len(orbits)):
        slices = []
        for j in range(settings.slices):
            orbit_slice = orbits[i][j*time_length:-(settings.slices-j)*time_length]
            slices.append(orbit_slice)
        orbit_collection.append(slices)
    orbit_collection = np.array(orbit_collection)
    orbit_collection = np.reshape(orbit_collection, (settings.number_of_curves, settings.timesteps))

    return orbit_collection

def basisfuns(A):
    """Generate basis vectors describing a set."""
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
    return lam,VT

def inner_product(matrix1, matrix2):
    """Take the inner product between two matrices."""
    c = np.zeros((matrix1.shape[0], matrix2.shape[0]), dtype=complex)
    for n in range(c.shape[0]):
        for m in range(c.shape[1]):
            c[n][m:m+1] = np.dot(matrix1[n], matrix2[m].conj().T)
    return c

def relativistic_components(phi_c, phi_nr):
    """Extract the relativistic components from the orbit differentials."""
    delta = 1*phi_c
    for n in range(len(delta)):
        for m in range(len(phi_nr)):
            delta[n] -= np.dot(delta[n],phi_nr[m].conj().T) * phi_nr[m]
    return delta

def plot_orbits(reference_orbit, rel_orbits, clas_orbits):
    """Plot the set of orbits and the set of orbit differentials."""
    # Set up plotting.
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect = 1.0)
    ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect = 1.0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Collection of orbits')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Orbit differentials')

    # Plot data.
    rel_differences, classical_differences, combined_differences = orbits.get_difference_vectors(
                                                        reference_orbit, rel_orbits, clas_orbits)
    for i in range(len(rel_orbits)):
        ax1.plot(rel_orbits[i].real, rel_orbits[i].imag, 'b')
        ax1.plot(clas_orbits[i].real, clas_orbits[i].imag, 'g')

        ax2.plot(combined_differences[i].real, rel_differences[i].imag)
        ax2.plot(classical_differences[i].real, classical_differences[i].imag)

    plt.show()

def generate_relativistic_basis(reference_orbit, rel_orbits, clas_orbits, settings):
    """Recreate original orbits from the relativistic components."""
    rel_differences, classical_differences, combined_differences = orbits.get_difference_vectors(
                                                        reference_orbit, rel_orbits, clas_orbits)
    # Generate differential orbit basis functions (phi_c, phi_nr)
    str_c, phi_c = basisfuns(np.matrix(combined_differences))
    str_nr, phi_nr = basisfuns(np.matrix(classical_differences))

    # Extract relativistic components of the differential orbit basis functions (psi)
    psi = relativistic_components(np.matrix(phi_c[:settings.cmpts]), np.matrix(phi_nr[:settings.cmpts])) #sliced at point where str goes to 0

    # Generate basis functions of the relativistic components
    # (should be orthonormal to the non-relativistic components) (psi_basis)
    strength, psi_basis = basisfuns(psi)

    # Recreate original orbits from the relativistic components (z from psi)
    # This is a projection of the orbits onto the psi basis
    # Applying basis reconstruction to the classical differences using psi
    # results only in the reference orbit because there are no relativistic components
    # whereas applying to relativistic difference orbits produces different curves
    basis_reconstruction = np.zeros((settings.number_of_curves, settings.timesteps), dtype=complex)
    for i in range(settings.number_of_curves):
        rel_dif = np.matrix(rel_differences[i])
        basis_reconstruction[i] = sum([(inner_product(rel_dif, psi_basis[n])*psi_basis[n]) for n in range(settings.cmpts)]) + reference_orbit

    return  basis_reconstruction


################################
############ METHOD ############
################################

# Create orbits (z), orbit differentials (z - z_ref)
settings = Settings()
orbits = Orbit_Solution(settings)
reference_orbit, rel_orbits, clas_orbits, unpeturbed_rel = orbits.get_orbits()
# Take time slices of orbits to produce full set
rel_orbits = timeslice(rel_orbits)
clas_orbits = timeslice(clas_orbits)

plot_orbits(reference_orbit, rel_orbits, clas_orbits)

# Reconstruct components of orbits that are purely relativistic and not found in the classical orbits
basis_reconstruction = generate_relativistic_basis(reference_orbit, rel_orbits, clas_orbits, settings)

#Plot reconstructed orbit with reference orbit, original orbit and original orbit as it would have looked had it not been peturbed
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, adjustable='box', aspect = 1.0)
ax1.plot(reference_orbit.real, reference_orbit.imag, label="Reference orbit")
ax1.plot(rel_orbits[0].real, rel_orbits[0].imag, label="Relativistic orbit with Newtonian perturber")
ax1.plot(basis_reconstruction[0].real, basis_reconstruction[0].imag, label="Reconstruction of relativistic orbit with Newtonian perturbation removed")
ax1.plot(unpeturbed_rel.real, unpeturbed_rel.imag, '--', label="Unpeturbed relativistic orbit")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.legend()
plt.show()


