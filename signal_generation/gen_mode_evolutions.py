# Generate frequencies, amplitudes for random mode(s)
import numpy as np
from Time_dep_Freq import gen_coupled_freq, debug_mode_frequency_plot
from scipy.signal import butter, filtfilt
import eqtools as eq
from freeqdsk import geqdsk
import os

###################33
def gen_mode_params(training_shots=1,params={'T': 10, 'dt': 0.01},doPlot=False,save_ext=''):

    # For now: hardcore frequency, amplitude evolution
    params_per_shot = []
    for _ in range(training_shots):
        # For now: Hardcode mode numers:
        params['m'] = [1, 6, 10]  # List of poloidal mode numbers
        params['n'] = [1, 2, 9]  # List of toroidal mode numbers
        # Eventually: more complex, randomly sampled evolution, potentially including AE bandgap analysis

        # Frequency, amplitude modulation
        # Note: If the amplitude and frequency are not set correctly for LF signals, 
        # the modulation frequency will dominate the spectrogram
        # Separately, if the noise envelope is too high, it induces some odd integration noise
        time = np.linspace(0,params['T'],int(params['T']/params['dt']))
        periods = 5
        dead_fraction = 0.4
        f_mod = lambda t: 7e3 + 3e3*t
        I_mod = lambda t: .1*4*(5 + 7*t**4)
        
        f_out_1, I_out_1, f_out_plot_1 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                        random_seed=42)

        
        periods = 2
        dead_fraction = 0.2
        f_mod = lambda t: 35e3 + 5e3*t
        I_mod = lambda t: .05*(6 + 3*np.sin(periods*2*np.pi*t))
        
        f_out_2, I_out_2, f_out_plot_2 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                            random_seed=42)

        periods = 3
        dead_fraction = 0.2
        f_mod = lambda t: 200e3 - 5e3*t**2
        I_mod = lambda t: .02*(2 + 4*t**2)
        
        f_out_3, I_out_3, f_out_plot_3 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                            random_seed=42)
        
        params['f'] = [f_out_1, f_out_2, f_out_3]  # List of frequencies for each m/n pair
        params['I'] = [I_out_1, I_out_2, I_out_3]  # List of amplitudes for each m/n pair

        params_per_shot.append(params.copy())  # Append the current shot's parameters

        if doPlot:
            # Plot the frequency modulation
            debug_mode_frequency_plot(time,f_out_1,I_out_1,f_out_plot_1,f_out_2,I_out_2,f_out_plot_2,f_out_3,\
                                  I_out_3,f_out_plot_3,save_ext)


    return params_per_shot # Params now include frequency bands, structured as list of dicts for each shot

######################################################################################################
def gen_mode_params_for_training(training_shots=1,\
        params={'T': 1e-3, 'dt': 1e-7,'m_pts':60,'n_pts':60,'periods':1,'R':None,'r':None,\
                   'noise_envelope':0.00,'n_threads':12},doPlot=True,save_ext=''):
    """
    Generate mode parameters for training.
    :param training_shots: Number of training shots
    :param params: Dictionary containing parameters like T and dt
    :param doPlot: Boolean to indicate if plots should be generated
    :param save_ext: Extension for saving plots
    :return: List of dictionaries with mode parameters for each shot
    """

    params_per_shot = [] # Empty list: will contain dicts for each shot

    # Load the coorect number of gEQDSK files
    gEQDSK_files = __build_geqdsks(training_shots,justLoad=False)
    # Loop over the number of training shots

    for shot in range(training_shots):
        # Reset params dict for each shot
        params = {'T':params['T'], 'dt':params['dt'],'m_pts':params['m_pts'],\
                  'n_pts':params['n_pts'],'periods':params['periods'],'R':params['R'],\
                    'r':params['r'],'noise_envelope':params['noise_envelope'],\
                        'n_threads':params['n_threads'],\
                            'max_m':params['max_m'],'max_n':params['max_n'],\
                                'max_modes':params['max_modes']} 

        # Randomly select a gEQDSK file, to determind minimum, maximum m/n
        params['file_geqdsk'] = 'gEQDSK_files/'+gEQDSK_files[shot]

         # Generate up to 5 modes per shot
        try:
            params['m'],params['n'] = __get_plausible_mn_values(\
                gEQDSK_file=params['file_geqdsk'],max_modes=params['max_modes'],max_n=params['max_n'],max_m=params['max_m'])
        except: continue  # If fail to get plausible m/n values, skip this shot
        
        params['f'] = []
        params['I'] = []
        params['f_Hz'] = []
        time = np.linspace(0,params['T'],int(params['T']/params['dt']))
        f_plot = []
        for n in params['n']:
            # Generate frequency evolution for each mode
            f_local = gen_frequency_evolutions(n, params['m'][params['n'].index(n)]) 
            # Generate amplitude evolution for each mode
            I_local = gen_amplitude_evolution(np.mean(f_local(np.array([0,1]))))  # Use the average frequency for amplitude scaling

            # Generate the time-dependent frequency and amplitude
            dead_fraction = np.random.uniform(0.1, 0.5) 
            periods = np.random.randint(1, int(1/dead_fraction))  # Randomly choose number of periods
            f_local, I_local, f_plot_local = gen_coupled_freq(time, periods, dead_fraction, f_local, I_local)

            params['f'].append(f_local)  # Append frequency evolution
            params['I'].append(I_local)  # Append amplitude evolution
            params['f_Hz'].append(f_plot_local) # f in Hz for plotting and training comparison
        
        params_per_shot.append(params.copy())  # Append the current shot's parameters

        if doPlot:
            # Plot the frequency modulation
            if len(params['f']) == 1:
                debug_mode_frequency_plot(time,params['f'][0],params['I'][0],params['f_Hz'][0],[],[],[],[],[],[],save_ext)
            elif len(params['f']) == 2:
                debug_mode_frequency_plot(time,params['f'][0],params['I'][0],params['f_Hz'][0],\
                                         params['f'][1],params['I'][1],params['f_Hz'][1],[],[],[],save_ext)
            elif len(params['f']) >= 3:
                debug_mode_frequency_plot(time,params['f'][0],params['I'][0],params['f_Hz'][0],\
                                         params['f'][1],params['I'][1],params['f_Hz'][1],\
                                         params['f'][2],params['I'][2],params['f_Hz'][2],save_ext)
    return params_per_shot # Params now include frequency bands, structured as list of dicts for each shot
######################################################################################################
def gen_frequency_evolutions(n, m, v_phi_0=[5, 30], v_a=[200, 400]):
    """
    Generates a time-dependent frequency evolution f(t) with a base frequency,
    a smooth evolution (linear or polynomial), and a smooth perturbation.

    Args:
        m (int): Poloidal mode number (not directly used in this function, but kept for consistency).
            Eventually, m could be used to determine position on V_phi(r~m/n)
        n (int): Toroidal mode number.
        v_phi_0 (list): Range for the base frequency component (kHz).
        v_a (list): Range for the base AE frequency component (kHz).
        T (float): Total time duration (seconds).
        dt (float): Time step (seconds).

    Returns:
        np.ndarray: Time array.
        np.ndarray: Frequency evolution f(t).
    """
    rng = np.random.default_rng()


    # 1. Base Frequency
    f_0 = n * rng.uniform(v_phi_0[0], v_phi_0[1]) * 1e3  # Approximate Doppler shifted frequency, in Hz
    if n >= 6:  # AE
        f_0 += rng.uniform(v_a[0], v_a[1]) * 1e3  # Proxy for base AE frequency, Hz

    # 2. Smooth Evolution (Linear or Polynomial)
    evolution_type = rng.choice(['linear'])#, 'polynomial'])
    if evolution_type == 'linear':
        slope = rng.uniform(-0.1, 0.1) * f_0  # Linear slope (fraction of base frequency)
        f_evol = lambda time: slope * time + f_0
    else:  # polynomial
        a = rng.uniform(-0.005, 0.005) * f_0  # Quadratic coefficient (fraction of base frequency)
        b = rng.uniform(-0.05, 0.05) * f_0 # Linear coefficient
        f_evol = lambda time: a * time**2 + b * time + f_0

    # 3. Smooth Perturbation (Sinusoidal or Random)
    perturbation_type = rng.choice(['sinusoidal', 'random'])
    if perturbation_type == 'sinusoidal':
        freq_mod = rng.uniform(1,6)  # Modulation frequency (Hz)
        amp_mod = rng.uniform(0.01, 0.01) * f_0  # Modulation amplitude (fraction of base frequency)
        f_pert = lambda time: amp_mod * np.sin(2 * np.pi * freq_mod * time)
    else:  # random
        rand_pert = lambda time: rng.normal(0, 0.001 * f_0, len(time))  # Random noise (fraction of base frequency)
        # Smooth the random perturbation using a low-pass filter
        #cutoff_freq = 5  # Cutoff frequency (Hz)
        # normalized_cutoff =  lambda time: cutoff_freq / (0.5 / (time[1]-time[0]))  # Nyquist frequency
        #ba = lambda time: butter(3, normalized_cutoff(time), btype='lowpass', analog=False)  # 3rd order Butterworth filter
        f_pert = lambda time: rand_pert(time)  # Apply filter (zero-phase)

    # 4. Combine Evolution and Perturbation
    f_t = lambda time: f_evol(time) + f_pert(time)

    return f_t

########################################################################################################
def gen_amplitude_evolution(freq_avg):
    """
    Generates a smooth, time-dependent amplitude evolution I(t).
    Args:
        freq_avg (float): Average frequency for the mode, used to scale the amplitude,
          such that I(t) * dt B(f(t)) ~ const acros modes
    Returns:
        
        function: Amplitude evolution I(t).
    """
    rng = np.random.default_rng()

    # 1. Base Amplitude
    I_0 = rng.uniform(1, 5) * 1/freq_avg  # Base amplitude, normalized by frequency

    # 2. Smooth Evolution (Linear or Polynomial)
    evolution_type = rng.choice(['linear'])#, 'polynomial'])
    if evolution_type == 'linear':
        slope = rng.uniform(-0.05, 0.05) * I_0  # Linear slope (fraction of base amplitude)
        I_evol = lambda time: slope * time + I_0
    else:  # polynomial
        a = rng.uniform(-0.001, 0.001) * I_0  # Quadratic coefficient (fraction of base amplitude)
        b = rng.uniform(-0.01, 0.01) * I_0  # Linear coefficient
        I_evol = lambda time: a * time**2 + b * time + I_0

    # 3. Smooth Perturbation (Sinusoidal or Random)
    perturbation_type = rng.choice(['sinusoidal', 'random'])
    if perturbation_type == 'sinusoidal':
        freq_mod = rng.uniform(0.5, 5)  # Modulation frequency (Hz)
        amp_mod = rng.uniform(0.01, 0.1) * I_0  # Modulation amplitude (fraction of base amplitude)
        I_pert = lambda time: amp_mod * np.sin(2 * np.pi * freq_mod * time)
    else:  # random
        rand_pert = lambda time: rng.normal(0, 0.01 * I_0, len(time))  # Random noise (fraction of base amplitude)
        # Smooth the random perturbation using a low-pass filter
        cutoff_freq = 2  # Cutoff frequency (Hz)
        #normalized_cutoff = lambda time: cutoff_freq / (0.5 / (time[1]-time[0]))  # Nyquist frequency
        #ba = lambda time: butter(3, normalized_cutoff(time), btype='lowpass', analog=False)  # 3rd order Butterworth filter
        #I_pert = lambda time: filtfilt(*ba(time), rand_pert(time))  # Apply filter (zero-phase)
        I_pert = lambda time: rand_pert(time)

    # 4. Combine Evolution and Perturbation
    I_t = lambda time: np.abs(I_evol(time) + I_pert(time))

    return I_t

######################################################################################################
def __get_plausible_mn_values(gEQDSK_file,max_n=15,max_m=15,max_modes=5):
    # Given a gEQDSK file, return a list of plausible m/n values
    # return as list of [m],[n] pairs
    # Load the gEQDSK file

    with open('input_data/'+gEQDSK_file,'r') as f: eqdsk=geqdsk.read(f)
    

    # Loop over possible m/n pairs, check if they are plausible
    plausible_mode_pairs = []
    for n in range(1,max_n+1):
        lower_m = np.ceil(eqdsk.qpsi[0]*n).astype(int);
        upper_m = np.floor(eqdsk.qpsi[-1]*n).astype(int)

        # Check for unreasonably high m values
        if upper_m > max_m: upper_m = max_m
        if upper_m > n*3: upper_m = n*3 
        if upper_m > n+4: upper_m = n+4

        for m in range(lower_m, upper_m+1):
            # Double check just in case:
            if eqdsk.qpsi[0]< m/n < eqdsk.qpsi[-1]: plausible_mode_pairs.append((m,n))

    plausible_mode_pairs = np.array(plausible_mode_pairs)
    out_mode_pairs =  plausible_mode_pairs[np.random.choice(len(plausible_mode_pairs),\
                                    size=np.random.randint(1,max_modes+1),replace=False).tolist() ]
    return out_mode_pairs[:,0].tolist(), out_mode_pairs[:,1].tolist() # 

######################################################################################################
def __build_geqdsks(n_equilibria,shot_list_file='../C-Mod/C_Mod_Shot_List_with_TAEs_Sheet1.csv', \
                    output_file='input_data/gEQDSK_files/',time_range=[.75,1.25],debug=True,
                    justLoad=False):

    # Only pulling from existing gEQDSK files, not loading new ones
    if justLoad:
        files = os.listdir(output_file)
        if len(files) < n_equilibria:
            raise ValueError(f"Requested {n_equilibria} equilibria, but only {len(files)} found in {output_file}.")
        else:
            return np.random.choice(files,n_equilibria,replace=False)
    
    #####################
    shotnos = np.loadtxt(shot_list_file,skiprows=1,delimiter=',',usecols=0,dtype=int)
    
    # +20% is saftey margin, some EFITs don't converge properly
    shots = np.random.choice(shotnos,int(n_equilibria*1.2),replace=True)
    times = np.random.choice(np.linspace(time_range[0],time_range[1],100+1,endpoint=True),int(n_equilibria*1.2),replace=True)

    g_files = ['g%d.%04d'%(shots[ind],times[ind]*1e3) for ind in range(n_equilibria)]

    g_files_out = [] # separate out file list, to account for gEQDSKs that fail to generate
    for ind,g_file in enumerate(g_files):
        if debug: print(f"Processing shot {shots[ind]} at time {times[ind]}: {g_file}")
        # Check if the gEQDSK file already exists
        if os.path.exists(output_file+g_file):
            g_files_out.append(g_file)
            continue

       
        try:
            eq_f = eq.CModEFIT.CModEFITTree(shots[ind])
            eq.filewriter.gfile(eq_f,times[ind],nw=200,nh=200,tunit='s',name=output_file+g_file)
            g_files_out.append(g_file)
        except Exception as e:
            if debug:
                print(f"Failed to generate gEQDSK for shot {shots[ind]} at time {times[ind]}: {e}")
            g_files_out.append(g_files_out[-1]) # Add previous file if fail to generate
            continue
    
    if len(g_files_out) < n_equilibria:
        raise ValueError(f"Requested {n_equilibria} equilibria, but only {len(g_files_out)} successfully generated.")
    
    return g_files_out


######################################################################################################
if __name__ == '__main__': 
    gen_mode_params_for_training()