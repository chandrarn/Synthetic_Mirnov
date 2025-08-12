# Generate frequencies, amplitudes for random mode(s)
import numpy as np
from Time_dep_Freq import gen_coupled_freq, debug_mode_frequency_plot

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
