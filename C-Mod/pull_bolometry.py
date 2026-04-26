# Basic code to pull AXUV diode and foil inversion profiles
# Data may not exist for all shots

from get_Cmod_Data import openTree
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def get_Emissivity(shotno=1110201006,doSaveData='',doPlotData=True,doSavePlots='', debug=True,\
                    emiss_lim=[0,1], bright_lim=[0,2], t_lim = [0, 2]):

    conn = openTree(shotno)

    # Zero-D signals:
    # dat = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.TWOPI_DIODE').data()

    # Get AXUV-A
    try:
        axa_emiss_profile = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXA.EMISS').data()
        axa_emiss_time = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXA.EMISS, 1)').data()
        axa_emiss_radius = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXA.EMISS, 0)').data()
        if debug:
            print('Got AXUV-A profile data')
    except:
        if debug:
            print('Failed to get AXUV-A profile data')

    try:
        axa_bright_profile = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXA.BRCHK').data()
        axa_bright_time = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXA.BRCHK, 1)').data()
        axa_bright_radius = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXA.BRCHK, 0)').data()
        if debug:
            print('Got AXUV-A brightness data')
    except:
        if debug:
            print('Failed to get AXUV-A brightness data')

    

    # Get AXUV-J
    try:
        axj_emiss_profile = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXJ.EMISS').data()
        axj_emiss_time = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXJ.EMISS, 1)').data()
        axj_emiss_radius = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXJ.EMISS, 0)').data()
        if debug:
            print('Got AXUV-J profile data')
    except:
        if debug:
            print('Failed to get AXUV-J profile data')

    try:
        axj_bright_profile = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXJ.BRCHK').data()
        axj_bright_time = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXJ.BRCHK, 1)').data()
        axj_bright_radius = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.DIODE.AXJ.BRCHK, 0)').data()
        if debug:
            print('Got AXUV-J brightness data')
    except:
        if debug:
            print('Failed to get AXUV-J brightness data')

    # Get foil emissivity
    try:
        foil_emiss_profile = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.FOIL.EMISSIVITY').data().T
        foil_emiss_time = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.FOIL.EMISSIVITY, 1)').data()
        foil_emiss_radius = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.FOIL.EMISSIVITY, 0  )').data()
        if debug:
            print('Got foil emissivity data')
    except:
        if debug:
            print('Failed to get foil emissivity data')

    try:
        foil_bright_profile = conn.get(r'\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.FOIL.BRCHK').data().T
        foil_bright_time = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.FOIL.BRCHK, 0)').data()
        foil_bright_radius = conn.get(r'dim_of(\CMOD::TOP.SPECTROSCOPY.BOLOMETER.RESULTS.FOIL.BRCHK, 1)').data()
        if debug:
            print('Got foil brightness data')
    except:
        if debug:
            print('Failed to get foil brightness data')
    

    if doPlotData:
        fig, ax = plt.subplots(2,3, figsize=(15,10), layout='constrained', sharex=True, sharey=True)
        levels_emiss = np.linspace(emiss_lim[0], emiss_lim[1], 20)
        levels_bright = np.linspace(bright_lim[0], bright_lim[1], 20)
        
        if 'axa_emiss_profile' in locals():
            im0 = ax[0,0].contourf(axa_emiss_time, axa_emiss_radius,  axa_emiss_profile.T*1e-6,\
                                    levels=levels_emiss, zorder=-5)
            # fig.colorbar(im0, ax=ax[0,0], label=r'AXUV-A Emissivity ($MW/m^2$?)')
        ax[0,0].set_title(rf'{shotno} AXUV-A')
        # ax[0,0].set_xlabel('Time (s)')
        ax[0,0].set_ylabel('Emissivity Radius [m]')
        if 'axa_bright_profile' in locals():
            im1 = ax[1,0].contourf(axa_bright_time, axa_bright_radius, axa_bright_profile.T*1e-6,\
                                    levels=levels_bright, zorder=-5)
            # fig.colorbar(im1, ax=ax[1,0], label=r'AXUV-A Brightness ($MW/m^2$?)')
            # ax[1,0].set_title('AXUV-A Brightness')
        ax[1,0].set_xlabel('Time [s]')
        ax[1,0].set_ylabel('Brightness Radius [m]')
        if 'axj_emiss_profile' in locals(): 
            im2 = ax[0,1].contourf(axj_emiss_time, axj_emiss_radius, axj_emiss_profile.T*1e-6, levels=levels_emiss, zorder=-5)
            # fig.colorbar(im2, ax=ax[0,2], label=r'AXUV-J Emissivity ($MW/m^2$?)')
        ax[0,1].set_title('AXUV-J')
            # ax[0,2].set_ylabel('Time (s)')
            # ax[0,2].set_xlabel('Radius (m?)')
        if 'axj_bright_profile' in locals():        
            im3 = ax[1,1].contourf(axj_bright_time, axj_bright_radius, axj_bright_profile.T*1e-6, levels=levels_bright, zorder=-5)
            # fig.colorbar(im3, ax=ax[1,0], label=r'AXUV-J Brightness ($MW/m^2$?)')
            # ax[1,0].set_title('AXUV-J Brightness')
        ax[1,1].set_xlabel('Time [s]')
            # ax[1,0].set_ylabel('Radius (m?)')
        if 'foil_emiss_profile' in locals():
            im4 = ax[0,2].contourf(foil_emiss_time, foil_emiss_radius, foil_emiss_profile*1e-6, levels=levels_emiss, zorder=-5)
            fig.colorbar(im4, ax=ax[0,2], label=r'Foil Emissivity [$MW/m^2$]')
        ax[0,2].set_title('Foil')
            # ax[1,2].set_xlabel('Time (s)')
            # ax[1,2].set_xlabel('Radius (m?)')
        if 'foil_bright_profile' in locals():
            im5 = ax[1,2].contourf(foil_bright_radius, foil_bright_time, foil_bright_profile*1e-6, levels=levels_bright, zorder=-5)
            fig.colorbar(im5, ax=ax[1,2], label=r'Foil Brightness [$MW/m^2$]')
            # ax[1,2].set_title('Foil Brightness')
        ax[1,2].set_xlabel('Time [s]')
            # ax[1,2].set_xlabel('Radius (m?)')

        ax[0,0].set_xlim(t_lim)
        for axi in ax.flatten():axi.set_rasterization_zorder(-1)

        if doSavePlots:
            plt.savefig(doSavePlots+f'bolometer_profiles_{shotno}.pdf', transparent=True)
            print(f'Saved bolometer profile plots to {doSavePlots+f"bolometer_profiles_{shotno}.pdf"}')
        plt.show(block=False)

    if doSaveData:
        dataset = xr.Dataset()
        if 'axa_emiss_profile' in locals():
            dataset['axa_emiss_profile'] = (('time', 'radius'), axa_emiss_profile)
            dataset['axa_emiss_time'] = (('time',), axa_emiss_time)
            dataset['axa_emiss_radius'] = (('radius',), axa_emiss_radius)
        elif 'axa_bright_profile' in locals():
            dataset['axa_bright_profile'] = (('time', 'radius'), axa_bright_profile)
            dataset['axa_bright_time'] = (('time',), axa_bright_time)
            dataset['axa_bright_radius'] = (('radius',), axa_bright_radius)
        if 'axj_emiss_profile' in locals():
            dataset['axj_emiss_profile'] = (('time', 'radius'), axj_emiss_profile)
            dataset['axj_emiss_time'] = (('time',), axj_emiss_time)
            dataset['axj_emiss_radius'] = (('radius',), axj_emiss_radius)
        elif 'axj_bright_profile' in locals():
            dataset['axj_bright_profile'] = (('time', 'radius'), axj_bright_profile)
            dataset['axj_bright_time'] = (('time',), axj_bright_time)
            dataset['axj_bright_radius'] = (('radius',), axj_bright_radius)
        if 'foil_emiss_profile' in locals():
            dataset['foil_emiss_profile'] = (('time', 'radius'), foil_emiss_profile)
            dataset['foil_emiss_time'] = (('time',), foil_emiss_time)
            dataset['foil_emiss_radius'] = (('radius',), foil_emiss_radius)
        elif 'foil_bright_profile' in locals():
            dataset['foil_bright_profile'] = (('time', 'radius'), foil_bright_profile)
            dataset['foil_bright_time'] = (('time',), foil_bright_time)
            dataset['foil_bright_radius'] = (('radius',), foil_bright_radius)
        dataset.attrs['shotno'] = shotno
        dataset.to_netcdf(doSaveData+f'bolometer_profiles_{shotno}.nc')

####################
if __name__ == '__main__':
    get_Emissivity(shotno=1110201006, doSaveData='data/'*False, doPlotData=True, doSavePlots='../output_plots/')
    print('Done')