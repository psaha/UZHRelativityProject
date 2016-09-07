# Plots classical and relativistic orbits for varying initial conditions

import numpy as np
import scipy.integrate
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt


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
    x = eqns[0]
    y = eqns[1]
    px = eqns[2]
    py = eqns[3]
    r = np.sqrt(x**2 + y**2)
    dx = px - (x*px + y*py)*2*x/r**3
    dpx = ((x*px + y*py)*2*px/r**3
                - ((1 - 2/r)**-2)*x/r**3
                - ((x*px + y*py)**2)*3*x/r**5)
    dy = py - (x*px + y*py)*2*y/r**3
    dpy = ((x*px + y*py)*2*py/r**3
                - ((1 - 2/r)**-2)*y/r**3
                - ((x*px + y*py)**2)*3*y/r**5)
    return [dx, dy, dpx, dpy]

def classical_derivatives(eqns, tau):
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
    r = np.sqrt(x**2 + y**2)
    dx = px
    dpx = -x/r**3
    dy = py
    dpy = -y/r**3
    return [dx, dy, dpx, dpy]


class Orbit_Solution(object):

    def __init__(self, initial_values, number_of_curves):
        n = 250000
        self.tau = np.linspace(0.0, n, 1000)
        self.init = InitialConditions(initial_values)
        self.number_of_curves = number_of_curves

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

        reference_orbit = np.array(self.solve(classical_derivatives, [0, 0, 0, 0]))
        reference_orbit = reference_orbit[0] + 1j*reference_orbit[1]

        rel_orbits = []
        clas_orbits = []

        for i in range(self.number_of_curves):
            deviations = [30*(i-(self.number_of_curves-1)/2), 0, 0, 0]
            r_orb = self.solve(relativistic_derivatives, deviations)
            c_orb = self.solve(classical_derivatives, deviations)

            rel_orbits.append(r_orb)
            clas_orbits.append(c_orb)

        rel_orbits = np.array(rel_orbits)[:,0] + 1j*np.array(rel_orbits)[:,1]
        clas_orbits = np.array(clas_orbits)[:,0] + 1j*np.array(clas_orbits)[:,1]

        return reference_orbit, rel_orbits, clas_orbits

    def get_difference_vectors(self):

        reference_orbit = self.get_orbits()[0]
        rel_orbits = self.get_orbits()[1]
        clas_orbits = self.get_orbits()[2]
        rel_differences = np.zeros_like(rel_orbits)
        classical_differences = np.zeros_like(clas_orbits)

        for i in range(self.number_of_curves):
            rel_differences[i] = rel_orbits[i] - reference_orbit
            classical_differences[i] = clas_orbits[i] - reference_orbit
        combined_differences = np.concatenate((rel_differences, classical_differences), axis=0)

        return rel_differences, classical_differences, combined_differences


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


################################
############ METHOD ############
################################

# Set up initial values: [x, y, px, py], number of orbits tested.
initial_values = [2000.0, 0.0, 0.0, 0.01]
number_of_curves = 5

# Create orbits (z), differential orbits (z - z_ref)
orbits = Orbit_Solution(initial_values, number_of_curves)
reference_orbit, rel_orbits, clas_orbits = orbits.get_orbits()
rel_differences, classical_differences, combined_differences = orbits.get_difference_vectors()

# Generate differential orbit basis functions (phi_c, phi_nr)
str_c, phi_c = basisfuns(np.matrix(combined_differences))
str_nr, phi_nr = basisfuns(np.matrix(classical_differences))

# Extract relativistic components of the differential orbit basis functions (psi)
psi = relativistic_components(np.matrix(phi_c), np.matrix(phi_nr))

# Generate basis functions of the relativistic components (should be orthonormal to the non-relativistic components) (psi_basis)
strength, psi_basis = basisfuns(psi)
#print(inner_product(psi_basis, phi_nr)) # = zeros! yay!

# Recreate original orbits from the relativistic components (z from psi)
for i in range(number_of_curves):
    thing = np.matrix(rel_differences[i])     #[0] + 1j*rel_differences[i][1])
    basis_recomposition = sum([(inner_product(thing, psi[n])*psi[n]) for n in range(number_of_curves)]) + (reference_orbit)
#complex numbers include y bit
    plt.plot(basis_recomposition.real, basis_recomposition.imag, 'b.')
    plt.plot(thing.real + reference_orbit.real, thing.imag + reference_orbit.imag, 'm.')
    plt.plot(rel_orbits[i].real, rel_orbits[i].imag, 'r.')
    plt.plot(clas_orbits[i].real, clas_orbits[i].imag, 'g.')
    plt.show()



# whole orbit rotated wrt orbit also initial time phase not known to us so some displacement of the time series maybe
#doing a preliminary clasical fit
# trim off the apocentre part a bit (off) as nothing happens far away and look closer at pericentre
# also momentum 
# after that add a perturber or something
# want to see if we can tell difference between class and rel orbit without having precise info anbout theese quantities - only approx info - want to see if we can tell the difference













#phi_c = np.array(phi_c)
#phi_nr = np.array(phi_nr)
#str_c = np.array(str_c)
#str_nr = np.array(str_nr)

#new_x = reference_orbit[0] + np.array(str_c[0]*phi_c[0].real)
#new_y = reference_orbit[1] + np.array(str_c[0]*phi_c[0].imag)
#new_x2 = reference_orbit[0] + np.array(str_nr[0]*phi_nr[0].real)
#new_y2 = reference_orbit[1] + np.array(str_nr[0]*phi_nr[0].imag)

#for i in range(phi_c.shape[0]): #NB
#    for j in range(phi_c.shape[1]):
#        phi_c[i][j] *= str_c[i]
#        phi_nr[i][j] *= str_nr[i]


#new_x = reference_orbit[0] + np.array(strength[0]*delta_basis[0].real)
#new_y = reference_orbit[1] + np.array(strength[0]*delta_basis[0].imag)
#plt.scatter(reference_orbit[0], reference_orbit[1])
#plt.scatter(new_x, new_y)
#plt.scatter(new_x2, new_y2)
#plt.show() #TOO SMALL - ISSUE WITH STRENGTHS OF VECTORS BEING LOST?? TODO

##################################
############ PLOTTING ############
##################################

def plot_orbits(rel_orbits, clas_orbits, combined_differences, classical_differences, number_of_curves):
    # Set up plotting.
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect = 1.0)
    ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect = 1.0)
    ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect = 1.0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax3.set_xlabel('px')
    ax3.set_ylabel('py')

    # Plot data.
    for i in range(len(rel_orbits)):
        ax1.plot(rel_orbits[i][0], rel_orbits[i][1], 'b')
        ax1.plot(clas_orbits[i][0], clas_orbits[i][1], 'g')
        ax2.plot(combined_differences[i][0], combined_differences[i][1], label=('R', 30*(i-(number_of_curves-1)/2)))
        ax2.plot(classical_differences[i][0], classical_differences[i][1], label=('C', 30*(i-(number_of_curves-1)/2)))
        ax3.plot(combined_differences[i][2], combined_differences[i][3], label=('R', 30*(i-(number_of_curves-1)/2)))
        ax3.plot(classical_differences[i][2], classical_differences[i][3], label=('C', 30*(i-(number_of_curves-1)/2)))

    ax2.legend()
    ax3.legend()
    plt.show()

def plot_basis_fns(phi_c, phi_nr):
    phi_c_x = phi_c.real
    phi_c_y = phi_c.imag
    phi_nr_x = phi_nr.real
    phi_nr_y = phi_nr.imag
    fig, ax = plt.subplots(nrows=1, ncols=phi_c_x.shape[0])
    fig2, ax2 = plt.subplots(nrows=1, ncols=phi_nr_x.shape[0])
    for i in range(phi_c_x.shape[0]):
        ax[i].scatter(phi_c_x[i,:], phi_c_y[i,:])
    for i in range(phi_nr_x.shape[0]):
        ax2[i].scatter(phi_nr_x[i,:], phi_nr_y[i,:])

    fig.suptitle('Basis functions for relativistic and classical orbits combined')
    fig2.suptitle('Classical basis functions')

    plt.show()

#plot_orbits(rel_orbits, clas_orbits, combined_differences, classical_differences, number_of_curves)
#plot_basis_fns(phi_c, phi_nr)

#plot_basis_fns(delta_basis, delta_basis)




