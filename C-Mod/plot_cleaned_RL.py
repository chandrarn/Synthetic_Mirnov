import csv
import re
from collections import defaultdict
from Emperical_RLC_C_mod import RLMeasurement, plot_RL_vs_frequency
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)

def load_cleaned_RL_csv(path):
    """
    Load cleaned CSV (sensor,freq,R_ohm,L_H) into {sensor: {freq: RLMeasurement}}
    """
    data = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["sensor"]
            freq = row["freq"]
            try:
                R = float(row["R_ohm"])
                L = float(row["L_H"])
            except Exception:
                continue
            data[name][freq] = RLMeasurement(R=R, L=L)
    return dict(data)

def plot_grouped_sensors(data):
    """
    Plot sensors grouped by pattern:
      - BPXX_ABK
      - BPXX_GHK
      - BPXT_ABK
      - BPXT_GHK
      - *_TOP or *_BOT (all together)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    patterns = {
        "BPXX_ABK": re.compile(r"^BP\d{2}_ABK$"),
        "BPXX_GHK": re.compile(r"^BP\d{2}_GHK$"),
        "BPXT_ABK": re.compile(r"^BP\dT_ABK$"),
        "BPXT_GHK": re.compile(r"^BP\dT_GHK$"),
        "AA_TOP_BOT": re.compile(r".*_(TOP|BOT)$"),
    }
    groups = {k: [] for k in patterns}

    for name in data:
        for k, pat in patterns.items():
            if pat.match(name):
                groups[k].append(name)
                break

    freq_keys = ["1k", "10k", "100k"]
    freqs = [1000, 10000, 100000]

    # Generate 30 distinct colors and line styles
    colors = cm.tab20(range(20)).tolist() + cm.tab20b(range(10)).tolist()
    line_styles = ['-', '--', '-.', ':']

    for group, sensors in groups.items():
        if not sensors:
            continue
        plt.figure(figsize=(10, 4))
        for idx, name in enumerate(sensors):
            freq_present = []
            R_vals = []
            L_vals = []
            for fk, f in zip(freq_keys, freqs):
                meas = data.get(name, {}).get(fk)
                if meas is not None:
                    freq_present.append(f*1e-3)
                    R_vals.append(meas.R)
                    L_vals.append(meas.L*1e6)
            if not freq_present:
                continue
            
            color = colors[idx % len(colors)]
            ls = line_styles[idx % len(line_styles)]
            
            plt.subplot(1, 2, 1)
            plt.plot(freq_present, R_vals, marker='o', label=name, color=color, linestyle=ls)
            plt.xscale('log')
            plt.xlabel('Frequency [kHz]')
            plt.ylabel('$\Omega$ [Ohms]')
            plt.title(f'{group} - Resistance')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(freq_present, L_vals, marker='o', label=name, color=color, linestyle=ls)
            plt.xscale('log')
            plt.xlabel('Frequency [kHz]')
            plt.ylabel(r'Inductance [$\mu$H]')
            plt.title(f'{group} - Inductance')
            plt.grid(True)
        plt.subplot(1, 2, 1)
        plt.legend(fontsize=8,handlelength=3,ncol=3,markerscale=.5)
        plt.subplot(1, 2, 2)
        # plt.legend(fontsize=8,handlelength=1,ncols=3)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"../output_plots/{group}_RL_plot.pdf", transparent=True)

if __name__ == "__main__":
    cleaned_data = load_cleaned_RL_csv("C_Mod_Calibrated_RLC_cleaned.csv")
    plot_grouped_sensors(cleaned_data)
    print("Plotted cleaned RL data.")
