###########################################################
#
#               HIREXSR
#
###########################################################
# import numpy as np
# import MDSplus
from header_Cmod import np, MDSplus

# Prepares necessary line_integrated brightness data
def prep_pos(
    line = None, 
    tht = None, 
    shot = None,
    ):
    # Line labels and central wavelengths
    lams = {'W': 3.94912, 'X': 3.96581, 'Z': 3.99417, 'LYA1': 3.73114, 'MO4D': 3.73980, 'J': 3.77179}
    lam0 = lams[line]
    specTree = MDSplus.Tree('spectroscopy', shot)
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS'
    # Loads THACO branch
    if tht > 0:
        rootPath += str(tht)
    rootPath += '.'

    # He-like branch
    if line in ['W', 'X', 'Z']:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'HELIKE')

        # pos vectors for detector modules 1-3 --- never used to look at Ca
        pos1 = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod1:pos'
                ).data()  # dim = (spectral pixel, spatial pixel, 4)
        pos2 = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod2:pos'
                ).data()  # dim = (spectral pixel, spatial pixel, 4)
        pos3 = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod3:pos'
                ).data()  # dim = (spectral pixel, spatial pixel, 4)

        # wavelengths for each module
        lam1 = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod1:lambda'
                ).data()  # dim = (spectral pixel, spatial pixel)
        lam2 = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod2:lambda'
                ).data()   # dim = (spectral pixel, spatial pixel)
        lam3 = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod3:lambda'
                ).data()  # dim = (spectral pixel, spatial pixel)

        pos_tot = np.hstack([pos1, pos2, pos3])  # dim = (spectral pixel, spatial pixel, 4)
        lam_tot = np.hstack([lam1, lam2, lam3])  # dim = (spectral pixel, spatial pixel)

    elif line in ['LYA1', 'MO4D', 'J']:
        # Loads Branch B data
        branchNode = specTree.getNode(rootPath+'HLIKE')

        # 1 detector module
        pos_tot =  specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod4:pos'
                ).data()   # dim = (spectral pixel, spatial pixel, 4)

        # wavelength
        lam_tot = specTree.getNode(
                r'\spectroscopy::top.hirexsr.calib.mod4:lambda'
                ).data()  # dim = (spectral pixel, spatial pixel)

    # Loads moment data
    momNode = branchNode.getNode(
        'MOMENTS.%s.MOM'%(line)
        )
    errNode = branchNode.getNode(
        'MOMENTS.%s.ERR'%(line)
        )

    # Integrated wavelength range
    try: 
        dlam = branchNode.getNode(
            'MOMENTS.%s.DLAM'%(line)
        ).data() / 1000  # [Ang]
    except: dlam = 5 / 1000 # 5nm in Ang, guess
    # Output
    return branchNode, momNode, errNode, pos_tot, lam_tot, lam0, dlam

'''
(
    branchNode, momNode, errNode, 
    pos_tot, lam_tot, lam0, dlam 
    ) = prep_pos(
        line= 'X',
        tht = dshots[shot]['tht'],
        shot = shot
        )
'''

# Loads geometrical data
def get_pos_XICS(
    line = None, 
    tht = None, 
    shot = None,
    ):

    # Init
    XICS = {}

    # Gets MDSplus tree data
    (
        branchNode, momNode, errNode, 
        pos_tot, lam_tot, lam0, dlam 
        ) = prep_pos(
            line= line,
            tht = tht,
            shot = shot
            )

    ##############################
    ### --- Loads POS data --- ###
    ##############################

    # Loads in channel mapping
    chmap = branchNode.getNode(
        'BINNING.CHMAP'
        ).data()[0]  # dim = (spatial pixel)
    maxChan = np.max(chmap) + 1

    # Bounds to integrate over
    lam_bounds = [lam0 - dlam, lam0 + dlam]  # [Ang]

    # Need to take into account the mapping spectral pixel->wavelength is a function of spatial pixel
    # Therefore, average POS vector over imaged wavelength range
    pos_specAve = np.zeros((len(chmap), 4))  # dim = (spatial pixel, 4)

    # Loop over spatial pixels
    for pixel in np.arange(len(chmap)):
        # Finds the bounds of the wavelength range considered
        bb = np.searchsorted(lam_tot[:, pixel], lam_bounds)

        # Averages POS vectors over wavelength range
        pos_specAve[pixel, :] = np.mean(pos_tot[bb[0] : bb[1], pixel, :], axis=0)

    # Average POS vector over the channel binning
    pos_ave = np.zeros((maxChan, 4))  # dim = (ch, 4)
    pos_std = np.zeros((maxChan, 4))  # dim = (ch, 4)
    for chord in np.arange(maxChan):
        # Averages POS vector over channel binning
        pos_ave[chord, :] = np.mean(pos_specAve[chmap == chord, :], axis=0)

    # Calculates the tangential Z-coordinate
    Z_T = pos_ave[:, 1] - np.tan(pos_ave[:, 3]) * np.sqrt(pos_ave[:, 0] ** 2 - pos_ave[:, 2] ** 2)  # [m], dim = n_ch

    # Calculates the toroidal angle subtended to the tangency radius
    phi_T = np.arccos(pos_ave[:, 2] / pos_ave[:, 0])  # [rad], dim = n_ch

    # Creates matrix to store LOS start point in Cartesian (x,y,z)-coordinates
    xyz_srt = np.stack((pos_ave[:, 0], 0 * pos_ave[:, 0], pos_ave[:, 1]), axis=1)  # dim = (n_ch, 3)

    # Creates matrix to store LOS end point in Cartesian (x,y,z)-coordinates
    X_T = pos_ave[:, 2] * np.cos(phi_T)
    Y_T = pos_ave[:, 2] * np.sin(phi_T)
    xyz_end = np.stack((X_T, Y_T, Z_T), axis=1)  # dim = (n_ch, 3)

    XICS['LOS_pt1'] = xyz_srt # dim(n_ch,3)
    XICS['LOS_pt2'] = xyz_end # dim(n_ch,2)

    # Output
    return XICS

##################################################################################
# Gets intensity time traces
def _get_int(
    dout={},
    ):

    # Initializes output
    dout['int'] = {}
    dout['vel'] = {}

    # Available data
    lines = ['A', 'M', 'W', 'Z']

    # MDSplus tree
    spec, _ = _cmod_tree(dout=dout)

    # Loop over available lines
    for ll in lines:
        # MDSplus node
        try:
            int_nd = spec.getNode(r'\spectroscopy::top.hirex_sr.analysis.'+ll+':int') 
            vel_nd = spec.getNode(r'\spectroscopy::top.hirex_sr.analysis.'+ll+':vel') 
        except:
            print(f'No data for line {ll}')
            continue
        # Stores data
        dout['int'][ll] = {}
        dout['int'][ll]['data'] = int_nd.data()
        dout['int'][ll]['t_s'] = int_nd.dim_of(0).data()

        dout['vel'][ll] = {}
        dout['vel'][ll]['data'] = vel_nd.data()
        dout['vel'][ll]['t_s'] = vel_nd.dim_of(0).data()

        # Stores branch
        if ll == 'A' or ll == 'M':
            dout['int'][ll]['branch'] = 'HLIKE'
            dout['vel'][ll]['branch'] = 'HLIKE'
        elif ll == 'W' or ll == 'Z':
            dout['int'][ll]['branch'] = 'HELIKE'
            dout['vel'][ll]['branch'] = 'HELIKE'

    # Output
    return dout
####################################################################
def plot_vel(dout={},intLim=[0,1e3], velLim=[-1e2,1e2]):
    # Plot the intensity and velocity traces for all lines
    import matplotlib.pyplot as plt
    lines = dout['int'].keys()
    fig, axs = plt.subplots(len(lines), 2, figsize=(6, 2*len(lines)), sharex=True,layout='constrained')
    for i, ll in enumerate(lines):  
        # Intensity
        # W, Z lines have three channels, A, M lines have one channel. Plot all channels if multiple
        npix = len(dout['int'][ll]['data']) if np.ndim(dout['int'][ll]['data']) > 1 else 1
        
        if npix == 1:
            axs[i, 0].plot(dout['int'][ll]['t_s'],dout['int'][ll]['data'])
            axs[i, 0].set_ylim(intLim[0], intLim[1])
        else:
            axs[i, 0].pcolormesh(dout['int'][ll]['t_s'],np.arange(npix),dout['int'][ll]['data'],\
                                 vmin=intLim[0], vmax=intLim[1], zorder=-10 )
            fig.colorbar(axs[i, 0].collections[0], ax=axs[i, 0], label='Intensity [arb. units]')

        # axs[i, 0].set_title(f'{ll} Intensity')
        if i == len(lines) - 1: axs[i, 0].set_xlabel('Time [s]')
        if npix ==  1: axs[i, 0].set_ylabel(f'{ll} Intensity [arb. units]')
        else: axs[i, 0].set_ylabel(f'{ll} Pixel Channel')

        # Velocity
        if npix == 1:
            axs[i, 1].plot(dout['vel'][ll]['t_s'], dout['vel'][ll]['data'])
            axs[i, 1].set_ylim(velLim[0], velLim[1])
        else:
            axs[i, 1].pcolormesh(dout['vel'][ll]['t_s'], np.arange(npix), dout['vel'][ll]['data'],\
                                 vmin=velLim[0], vmax=velLim[1], zorder=-10)
            fig.colorbar(axs[i, 1].collections[0], ax=axs[i, 1], label='Velocity [km/s]')



        # axs[i, 1].set_title(f'{ll} Velocity')
        if i == len(lines) - 1: axs[i, 1].set_xlabel('Time [s]')
        if npix == 1:
            axs[i, 1].set_ylabel(f'{ll} Velocity [km/s]')
        else:
            pass#axs[i, 1].set_ylabel(f'{ll} Pixel Channel')

        axs[i, 0].grid()
        axs[i, 1].grid()
    

    # fig.suptitle(f'Shot {dout["shot"]} - THT {dout["tht"]}', fontsize=16)
    axs[0,0].legend([f'Shot {dout["shot"]}'],fontsize=8,handlelength=1)
    # plt.tight_layout()
    for ax in axs.ravel():ax.set_rasterization_zorder(-1)

    fig.savefig(f'../output_plots/shot_{dout["shot"]}_tht_{dout["tht"]}_int_vel.pdf', transparent=True)
    print(f'Saved: ../output_plots/shot_{dout["shot"]}_tht_{dout["tht"]}_int_vel.pdf')
    plt.show()
############################################################################
# Gets MDSplus paths
def _cmod_tree(
    dout={},
    branch=None,
    ):

    # Obtains data
    specTree = MDSplus.Tree('spectroscopy', dout['shot'])
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS'

    # Loads THACO branch
    if dout['tht'] > 0:
        rootPath += str(dout['tht'])
    rootPath += '.'

    # Analysis branch
    if branch is not None:
        branchNode = specTree.getNode(rootPath+branch)
    else:
        branchNode = None

    # Output
    return specTree, branchNode


#########################################################################


# Init
dXICS = {}
kks = ['Z', 'LYA1']
shot = 1140701023#1051202011#
vel_out = {}
for kk in kks:
    # dXICS[kk] = get_pos_XICS(
    #     line = kk,
    #     tht = 0,#dshots[shot]['tht'],
    #     shot = shot
    #     )

    vel_out= _get_int(dout={'shot': shot, 'tht': 0})
    
plot_vel(dout=vel_out)
print('XICS LOS geometry data loaded')
# print(dXICS)
