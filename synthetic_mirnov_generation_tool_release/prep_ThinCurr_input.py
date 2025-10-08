#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from header_synthetic_mirnov_generation import np, EquilibriumField, EquilibriumFilament,\
    save_sensors,Mirnov, Fraction, geqdsk, plt, TraceType

################################################################################################
################################################################################################
def gen_coil_currs(param,debug=True,doSave=False,wind_in='theta',scan_in_freq=False):
    # Assign currents to existing fillaments for time dependent calculation
    # The ordering of the filaments and currents must match for the field topology to be correct

    m_pts=param['m_pts'];m=param['m'];dt=param['dt'];f=param['f'];periods=param['periods']
    noise_envelope = param['noise_envelope'] # Random noise envelope
    if type(m) is int: m = [m]# Convert type to correct format

    I=param['I'];n=param['n'];n_pts=param['n_pts']
    nsteps = int(param['T']/dt)
    
    # Need to get the list of starting angles, to correctly assign filament colors
    starting_angle,_= gen_filament_coords(param,wind_in=wind_in) # returns as lists
    time=np.linspace(0,param['T'],nsteps)
    
    # assume that m_pts can be a list
    if type(m_pts) == int: num_filaments = m_pts*len(starting_angle)
    else: num_filaments = sum([len(theta) for theta in starting_angle])
    if debug:print('Number of filaments: %d'%num_filaments)
    coil_currs = np.zeros((time.size,num_filaments+1))

    # Time vector in first column
    coil_currs[:,0]=time
    if debug:print('Initialized coil currents array with shape (%d,%d)'%coil_currs.shape)

#########################################################################################################
def gen_filament_coords(params, wind_in='phi'):
    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate theta,phi coordinates for fillaments
    # The points launch in a fractional sector of the poloidal or toroidal plane
    # (depending on 'wind_in' parameter), and
    # wrap enough times to return to their starting point

    starting_angle=[]; winding_angle=[]
    if type(m_pts) is int: m_pts = [m_pts]*len(m)
    if len(m_pts) != len(m): m_pts = m_pts*len(m)
    if type(n_pts) is int: n_pts = [n_pts]*len(m_pts)
    if len(n_pts) != len(m): n_pts = n_pts*len(m)
    
    for ind_m, m_ in enumerate(m if type(m) is list else [m]):
        
        n_ = n[ind_m] if type(n) is list else n
        ratio = Fraction(m_,n_)
        m_local=ratio.numerator;n_local=ratio.denominator
        if wind_in == 'phi':
            starting_angle.append(np.linspace(0,2*np.pi/m_local*n_local,m_pts[ind_m],endpoint=True))
            winding_angle.append(np.linspace(0,m_local*2*np.pi,n_pts[ind_m],endpoint=True))
        elif wind_in == 'theta':
            starting_angle.append( np.linspace(0,2*np.pi/n_local,n_pts[ind_m],endpoint=False) )
            winding_angle.append( np.linspace(0,2*np.pi*m_local,m_pts[ind_m]) )

    return starting_angle, winding_angle

#########################################################################################################
##########################################################################################
#########################################################################
#########################################################################
# Field Line Following Model
##########################################################################
#########################################################################

#########################################################################
# Initial phi positions for filaments to launch from
def starting_phi(m,n,m_pts,n_pts):
    ratio = Fraction(m,n)
    m_local=ratio.numerator;n_local=ratio.denominator
    phi_start = np.linspace(0,2*np.pi/n_local,n_pts,endpoint=False)
    phi_advance = np.linspace(0,2*np.pi*m_local,m_pts)
    return phi_start, phi_advance

###############################################################
# Core of new Winding Method
def wind_in_theta(file_geqdsk,m,n, debug=False):
    with open('input_data/'+file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
    eq_field = EquilibriumField(eqdsk)
    eq_filament = EquilibriumFilament(m,n,eq_field)
    filament_points,filament_etas = eq_filament._trace(method=TraceType.AVERAGE,num_filament_points=300)

    if debug:
        plt.close('Winding_Theta')
        plt.figure(num='Winding_Theta',tight_layout=True);
        plt.plot(filament_points[:,1]/(2*np.pi));
        plt.plot([0,len(filament_points)],[0,filament_points[-1,1]/(2*np.pi)],'--k');
        plt.grid();
        plt.ylabel(r'$\hat{\phi}$ [2$\pi$ Rad]')
        plt.xlabel('Filament Points')
        plt.show()

    # Nan Check
    return filament_points,filament_etas

#######################################
def conv_theta_wind_to_coords(filament_points,phi_start):
    # Convert R, Phi, Z for each filament, with starting launch offset in phi
    coords = []

    for phi_0 in phi_start:
        phi_ = filament_points[:,1]+phi_0 # local phi for a given filament
        X = filament_points[:,0]*np.cos(phi_)
        Y = filament_points[:,0]*np.sin(phi_)
        Z = filament_points[:,2]
        coords.append([X,Y,Z])
    
    return np.array(coords).T

def calc_filament_coords_field_lines(params,file_geqdsk,doDebug=False):
    # Calling container for field-tracing filaments

    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate theta,phi coordinates for fillaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point

    starting_angle=[]; winding_angle=[]
    if type(m_pts) is int: m_pts = [m_pts]*len(m)
    if type(n_pts) is int: n_pts = [n_pts]*len(m_pts)
    coords = []
    for ind_m, m_ in enumerate(m if type(m) is list else [m]):
        phi_start, phi_advance =  starting_phi(m_,n[ind_m],m_pts[ind_m],n_pts[ind_m])

        filament_points,filament_etas = wind_in_theta(file_geqdsk,m_,n[ind_m])

        coords_ = conv_theta_wind_to_coords(filament_points,phi_start)

        # if doDebug: debug_filament_currents(coords_,phi_start,n[ind_m])
        #coords_[:,0]= coords[:,1]
        # coords_[0,:2,:]=coords_[1,:2,:]
        coords.append(coords_.T)

    return coords


################################################################################################
################################################################################################
# Generate OFT Input Files
################################################################################################
def gen_OFT_filement_and_eta_file(filament_file,filament_coords,\
                  eta = '1.8E-5, 3.6E-5, 2.4E-5'):
    # Write filament (x,y,z) coordinates to xml file in OFT format 

    # Coords comes in as a list of m/n pairs, each with a list of filaments
    # Each filament is a list of points, each point is a list of [x,y,z] points
    # This holds even if there's only one m/n pair

    with open('input_data/'+filament_file,'w+') as f:
        f.write('<oft>\n\t<thincurr>\n\t<eta>%s</eta>\n\t<icoils>\n'%eta)
        
        print('File open for writing: %s'%filament_file)

        for filament_ in filament_coords:
            
            for filament in filament_:
                f.write('\t<coil_set>\n')
                f.write('\n\t\t<coil npts="%d" scale="1.0">\n'%np.shape(filament)[1])

                for xyz in np.array(filament).T:
                    x=xyz[0];y=xyz[1];z=xyz[2]
                    f.write('\t\t\t %1.3f, %1.3f, %1.3f\n'%(x,y,z) )

                f.write('\t\t</coil>\n')
                f.write('\t</coil_set>\n')

        f.write('\t</icoils>\n\t</thincurr>\n</oft>')

################################################################
#######3#########################################################
def gen_OFT_sensors_file(probe_details, working_files_directory, debug=True):
    # Assume probe_details is an xarray dataset with the following variables:
    # X, Y, Z (coordinates of each probe)
    # theta, phi (orientation of each probe)

    sensor_list = []
    for probe in probe_details.sensor_name.values:
        theta_pol = probe_details.theta.sel(sensor_name=probe).item() # assume degrees
        phi = np.arctan2(probe_details.Y.sel(sensor_name=probe).item(),\
                         probe_details.X.sel(sensor_name=probe).item())
        pt = ( probe_details.X.sel(sensor_name=probe).item(),
                probe_details.Y.sel(sensor_name=probe).item(),
                probe_details.Z.sel(sensor_name=probe).item(),)
        # normal vector does not current account for toroidal tilt
        norm = [np.cos(theta_pol*np.pi/180)*np.cos(phi),
                np.cos(theta_pol*np.pi/180)*np.sin(phi),
                np.sin(theta_pol*np.pi/180)]
        # Probe radius
        dx = probe_details.dx.sel(sensor_name=probe).item()
        # create Mirnov object
        sensor_list.append(
            Mirnov(pt, norm, probe, dx)
        )
    
    # Save in ThinCurr format
    save_sensors(sensor_list, \
                 f'{working_files_directory}/floops_{probe_details.attrs['probe_set_name']}.loc')
    if debug: print('Wrote OFT sensor file to %s/floops_%s.loc'%(working_files_directory,probe_details.attrs['probe_set_name']))
    return sensor_list
