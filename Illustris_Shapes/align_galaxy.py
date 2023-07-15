import numpy as np

from Illustris_Shapes.calculate_galaxy_shapes import format_particles, particle_selection, galaxy_center
from Illustris_Shapes.calculate_galaxy_shapes import gas_selection, gas_selection_radtot
from illustris_python.snapshot import loadHalo, snapPath, loadSubhalo
from illustris_python.groupcat import gcPath, loadHalos, loadSubhalos
from rotations.rotations3d import rotation_matrices_from_basis
from rotations import rotate_vector_collection

# specify simulation
simname = 'TNG50-1'
snapNum = 99
# shape_type = 'iterative'
shape_type = 'reduced'

#load simulation information
from Illustris_Shapes.simulation_props import sim_prop_dict
d = sim_prop_dict[simname]
basePath = d['basePath']
Lbox = d['Lbox']
h = 0.6774
acosmo = 1

# load pre-computed shape catalog
from astropy.table import Table
fname = simname + '_' + str(snapNum) + '_' + shape_type + '_' +'galaxy_shapes_1e08'+ '.dat'
t_1 = Table.read('/scratch1/TNG/TNG50-1/data/shape_catalogs/'+fname, format='ascii')
t_2 = Table.read('/scratch1/projects/TNG/B/data/catalog_stellar_mass_greater_than_1e8_fdisk0.2_and_flatness0.7.txt',format='ascii')


##############
def axis_aligned_ptcl_dist(gal_id, Av, Bv, Cv, scale):
    """
    return the axis aligned stellar particle distribution of a galaxy
    """
    fields = ['SubhaloGrNr', 'SubhaloMassInRadType', 'SubhaloPos', 'SubhaloHalfmassRadType', 'SubhaloHalfmassRad']
    galaxy_table = loadSubhalos(basePath, snapNum, fields=fields)
    gal_position = galaxy_center(gal_id, galaxy_table)
    
    
    # load stellar particle positions and masses
    ptcl_coords = loadSubhalo(basePath, snapNum, gal_id, 4, fields=['Coordinates'])/1000.0 # cMpc/h
    ptcl_masses = loadSubhalo(basePath, snapNum, gal_id, 4, fields=['Masses'])*10**10/h    # M0
    print('Total stellar mass [10^10 M0/h] = ', np.sum(ptcl_masses)/1e10)
    
    # load gas cells positions, masses and B
    gas_coords = loadSubhalo(basePath, snapNum, gal_id, 0, fields=['Coordinates'])/1000.0 # cMpc/h
    gas_masses = loadSubhalo(basePath, snapNum, gal_id, 0, fields=['Masses'])*10**10/h    # M0
    Bvec       = loadSubhalo(basePath, snapNum, gal_id, 0, fields=['MagneticField'])#*h/(acosmo**2)*2.6 # micro Gauss
#     rho        = loadSubhalo(basePath, snapNum, gal_id, 0, fields=['Density'])#*10**10/h / ((h**(-3)) # M0/(kpc^3)
    rho        = loadSubhalo(basePath, snapNum, gal_id, 0, fields=['Density'])*10**10*h*h # M0/(kpc^3)

    
    print('Total gas mass [10^10 M0/h] = ', np.sum(gas_masses)/1e10)
    print('N gas cells', gas_masses.shape)

    # center and account for PBCs
    ptcl_coords = format_particles(gal_position, ptcl_coords, Lbox)
    gas_coords  = format_particles(gal_position, gas_coords, Lbox)
    
    # make selection
    ptcl_mask = particle_selection(gal_id, ptcl_coords, galaxy_table,
                                   basePath, snapNum, radial_mask=True)
    gas_mask  = gas_selection(gal_id, gas_coords, galaxy_table,
                                   basePath, snapNum, radial_mask=True)
    
    # extract particles and cells for the galaxy
    ptcl_coords = ptcl_coords[ptcl_mask]
    ptcl_masses = ptcl_masses[ptcl_mask]
    gas_coords  = gas_coords[gas_mask]
    gas_masses  = gas_masses[gas_mask]
    Bvec        = Bvec[gas_mask]
    rho         = rho[gas_mask]
    print('N gas cells after radial mask', gas_masses.shape)
    
    # build a rotation matrix
    rot = rotation_matrices_from_basis([Av], [Bv], [Cv])
    rot = rot[0].T
    
    # rotate the particles and cells to be axis-aligned
    ptcl_coords = rotate_vector_collection(rot, ptcl_coords)
    gas_coords  = rotate_vector_collection(rot, gas_coords)
#     Bvec        = rotate_vector_collection(rot, Bvec)
    
    # get the half mass radius
    gal_rhalfs = loadSubhalos(basePath, snapNum, fields=['SubhaloHalfmassRadType'])[:,4]/1000.00 # cMpc/h
    SubhaloHalfmassRad = loadSubhalos(basePath, snapNum, fields=['SubhaloHalfmassRad'])[:]/1000.00 # cMpc/h
    gal_rhalf = gal_rhalfs[gal_id]
    scaleH = 1./np.e*gal_rhalf

    print("Stellar effective radius of", gal_id, "=", 2*gal_rhalf*1000/h, "kpc")
    print("Total effective radius of", gal_id, "=", 2*SubhaloHalfmassRad[gal_id]*1000/h, "kpc")
    
    rcyl           = np.sqrt(ptcl_coords[:,0]**2 + ptcl_coords[:,1]**2)
    mask_out_of_re = (rcyl < (1*gal_rhalf))
    ptcl_coords    = ptcl_coords[mask_out_of_re]
    ptcl_masses    = ptcl_masses[mask_out_of_re]
    
    rcyl           = np.sqrt(gas_coords[:,0]**2 + gas_coords[:,1]**2)
    mask_out_of_re = (rcyl < (1*gal_rhalf))
    gas_coords     = gas_coords[mask_out_of_re]
    gas_masses     = gas_masses[mask_out_of_re]
    Bvec           = Bvec[mask_out_of_re]
    rho            = rho[mask_out_of_re]
    
    mask_ptcl_out_of_z = (np.abs(ptcl_coords[:,2]*1000/h) < scaleH*1000/h)
    ptcl_coords        = ptcl_coords[mask_ptcl_out_of_z]
    ptcl_masses        = ptcl_masses[mask_ptcl_out_of_z]
    
    mask_gas_out_of_z = (np.abs(gas_coords[:,2]*1000/h) < scaleH*1000/h)
    gas_coords        = gas_coords[mask_gas_out_of_z]
    gas_masses        = gas_masses[mask_gas_out_of_z]
    Bvec              = Bvec[mask_gas_out_of_z]
    rho               = rho[mask_gas_out_of_z]
    
    
    # scale coordinates by the half-mass radius
    if scale == True:
        return ptcl_coords/gal_rhalf, ptcl_masses, gas_coords/gal_rhalf, gas_masses, Bvec, rho
    elif scale == False:
        return ptcl_coords,           ptcl_masses, gas_coords,           gas_masses, Bvec, rho