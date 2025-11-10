"""
Empirical RLC helpers for C-Mod Mirnov coils.

Features:
 - Load calibrated R/L measurements from a spreadsheet-like CSV
 - Ignore header row and trailing comment column
 - Parse R/L pairs at 1kHz, 10kHz, 100kHz (not all present for every sensor)
 - Drop individual measurements outside user-configurable thresholds
 - Handle duplicate sensor names by averaging valid measurements per frequency
 - Provide a clean dictionary output and optional CSV writer

Expected CSV columns (by position):
 0: sensor_name, 1: R@1k, 2: L@1k, 3: R@10k, 4: L@10k, 5: R@100k, 6: L@100k, 7: comments (ignored)

Notes:
 - If either R or L for a frequency is outside threshold, that (sensor, freq) sample is ignored.
 - If a sensor appears multiple times, valid values for the same frequency are averaged.
 - Thresholds default to conservative ranges and can be overridden via function args or CLI.
"""

import csv
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()
from matplotlib import rc
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
import numpy as np
import re
from collections import defaultdict


# Default thresholds (adjust as needed for C-Mod coils)
DEFAULT_R_MIN = 0.05   # Ohm
DEFAULT_R_MAX = 50.0  # Ohm
DEFAULT_L_MIN = 3e-5   # H
DEFAULT_L_MAX = 1e-4   # H

FREQUENCIES_HZ = {
    "1k": 1_000.0,
    "10k": 10_000.0,
    "100k": 100_000.0,
}


@dataclass
class RLMeasurement:
    R: float  # Ohm
    L: float  # H
    C: float  # F


def _safe_float(val: str) -> Optional[float]:
    """Parse float from a string; return None on empty/invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "-", "--"}:
        return None
    try:
        # Keep only characters that can appear in float literals
        filtered = "".join(ch for ch in s if ch.isdigit() or ch in ".eE-+")
        if filtered == "":
            return None
        return float(filtered)
    except Exception:
        return None


def _in_thresholds(R: Optional[float], L: Optional[float],
                   R_min: float, R_max: float, L_min: float, L_max: float) -> bool:
    """Return True if both R and L are present and within thresholds."""
    if R is None or L is None:
        return False
    return (R_min <= R <= R_max) and (L_min <= L <= L_max)


def load_calibrated_RL_csv(
    csv_path: str,
    R_min: float = DEFAULT_R_MIN,
    R_max: float = DEFAULT_R_MAX,
    L_min: float = DEFAULT_L_MIN,
    L_max: float = DEFAULT_L_MAX,
    L_scale: float = 1e-6,
) -> Dict[str, Dict[str, RLMeasurement]]:
    """
    Load a CSV of calibrated R/L values and return averaged measurements per sensor and frequency.

    Input CSV format (positional columns):
        0 sensor_name
        1 R@1k, 2 L@1k, 3 R@10k, 4 L@10k, 5 R@100k, 6 L@100k, 7 comments (ignored)
    The first row is headings and will be ignored.

    Returns a dict: { sensor_name: { freq_key: RLMeasurement(R, L), ... }, ... }
        where freq_key in {"1k", "10k", "100k"}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Temporary storage for aggregating duplicates
    sums: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    # maps sensor -> freq_key -> (sum_R, sum_L, count)

    with open(csv_path, "r", newline="") as f:
        lines = f.read().splitlines()

    if not lines:
        return {}

    # Skip header line
    data_lines = lines[1:]

    # Heuristic section tracking: start assuming first block is 1k, then 10k, then 100k
    freq_order = ["1k", "10k", "100k"]
    section_idx = 0

    # Track how many pairs per line we are typically seeing to infer section boundaries
    recent_pair_counts: list[int] = []

    # Also track which frequencies have already been set per sensor to place later single-pair entries
    seen_freqs: Dict[str, set] = {}

    for line_idx, line in enumerate(data_lines, start=2):
        raw = line.rstrip()
        if not raw:
            # Blank line: consider this a potential section separator
            if recent_pair_counts:
                # If we recently saw mostly single-pair lines, advance section
                if sum(1 for c in recent_pair_counts[-10:] if c == 1) >= 7:
                    section_idx = min(section_idx + 1, len(freq_order) - 1)
            recent_pair_counts.clear()
            continue

        parts = raw.split()
        if not parts:
            continue

        name = parts[0]
        num_tokens = []
        # Collect numeric-like tokens; tolerate concatenated forms like 1200-1.73E+007 by splitting on pattern
        for tok in parts[1:]:
            # Attempt to split concatenated number pairs (e.g., '1200-1.73E+007')
            # Heuristic: if token contains both digits and a +/- not at start, try split at the last sign
            if ("+" in tok[1:] or "-" in tok[1:]) and any(ch.isdigit() for ch in tok):
                # Try split on last '+' or '-' occurrence that's not at position 0
                split_pos = max(tok.rfind("+"), tok.rfind("-"))
                if split_pos > 0:
                    first = tok[:split_pos]
                    second = tok[split_pos:]
                    for sub in (first, second):
                        val = _safe_float(sub)
                        if val is not None:
                            num_tokens.append(val)
                    continue
            val = _safe_float(tok)
            if val is not None:
                num_tokens.append(val)

        # Group numeric tokens into pairs (R, L)
        pairs: list[Tuple[Optional[float], Optional[float]]] = []
        for i in range(0, len(num_tokens), 2):
            if i + 1 < len(num_tokens):
                R_val = num_tokens[i]
                L_val = num_tokens[i + 1]
                pairs.append((R_val, L_val))

        if not pairs:
            continue

        recent_pair_counts.append(len(pairs))

        # Map pairs to frequencies
        if len(pairs) >= 3:
            mapping = {"1k": pairs[0], "10k": pairs[1], "100k": pairs[2]}
        elif len(pairs) == 2:
            # Assume order 1k, 10k for two pairs
            mapping = {"1k": pairs[0], "10k": pairs[1]}
        else:  # len == 1
            # Place the single pair at the most plausible next frequency for this sensor
            already = seen_freqs.get(name, set())
            # Prefer current section frequency if not already filled
            fk_guess = freq_order[section_idx]
            for fk in (fk_guess, "1k", "10k", "100k"):
                if fk not in already:
                    mapping = {fk: pairs[0]}
                    break
            else:
                mapping = {fk_guess: pairs[0]}

        # Apply thresholds and aggregate
        for fk, (R_val, L_val) in mapping.items():
            # Scale L from microhenry to henry by default (L_scale)
            L_scaled = L_val * L_scale if L_val is not None else None
            if not _in_thresholds(R_val, L_scaled, R_min, R_max, L_min, L_max):
                continue
            sums.setdefault(name, {})
            sum_R, sum_L, cnt = sums[name].get(fk, (0.0, 0.0, 0))
            sums[name][fk] = (sum_R + float(R_val), sum_L + float(L_scaled), cnt + 1)
            seen = seen_freqs.setdefault(name, set())
            seen.add(fk)

    # Convert sums to averages
    result: Dict[str, Dict[str, RLMeasurement]] = {}
    for name, freq_map in sums.items():
        out_freqs: Dict[str, RLMeasurement] = {}
        for fk, (sum_R, sum_L, cnt) in freq_map.items():
            if cnt <= 0:
                continue
            out_freqs[fk] = RLMeasurement(R=sum_R / cnt, L=sum_L / cnt)
        if out_freqs:
            result[name] = out_freqs

    return result


def write_cleaned_RL_csv(
    data: Dict[str, Dict[str, RLMeasurement]],
    out_path: str,
) -> None:
    """Write the cleaned/averaged data to a CSV with columns: name, freq, R, L."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "freq", "R_ohm", "L_H"])
        for name, freq_map in sorted(data.items()):
            # sort by the intended freq order 1k, 10k, 100k
            order = {k: i for i, k in enumerate(FREQUENCIES_HZ.keys())}
            for fk, meas in sorted(freq_map.items(), key=lambda kv: order.get(kv[0], 999)):
                w.writerow([name, fk, f"{meas.R:.6g}", f"{meas.L:.6g}"])

def generate_RLC_Transfer_Function_C_mod(plot_R=False):
    # Placeholder for future integration with per-sensor R/L
    # Load in empirical measurements (use load_calibrated_RL_csv elsewhere)
    return __prep_RLC_Transfer_Function(plot_R=plot_R)

def load_cleaned_RL_csv_simple(path):
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
                C = float(row["C_pF"])
            except Exception:
                continue
            data[name][freq] = RLMeasurement(R=R, L=L, C=C)
    return dict(data)

def __prep_RLC_Transfer_Function(
    cleaned_csv="C_Mod_Calibrated_RLC_cleaned.csv",
    C_0=760e-12,
    plot_R=False,
    R_0=0.7,
    bad_sensor_map={'BP21_ABK':['BP06_ABK'],'BP13_GHK':['BP02_GHK'],'BP6T_GHK':['BP1T_GHK'],\
                    'BP2T_GHK':['BP1T_GHK'],'BP5T_GHK':['BP1T_GHK']},
    missing_sensor_map={'BP4T_GHK':'BP3T_GHK'}, 
    good_sensor_map=None,
):
    """
    Returns a dict: {sensor: {"mag_RLC": mag_RLC_lambda, "phase_RLC": phase_RLC_lambda}}
    For BPXX_ABK/BPXX_GHK: uses average R/L and Z_R(w) correction.
    For others: fits R(f), L(f) if >1 point, else uses mean.
    For bad sensors: copies mag_RLC and phase_RLC from first available good sensor.
    """
    import numpy as np

    # Patterns for BPXX_ABK and BPXX_GHK
    pat_abk = re.compile(r"^BP\d{2}_ABK$")
    pat_ghk = re.compile(r"^BP\d{2}_GHK$")

    # Load cleaned data
    data = load_cleaned_RL_csv_simple(cleaned_csv)

    # Optionally specify bad/good sensors
    if bad_sensor_map is None:
        bad_sensor_map = {}  # e.g. {"BP15_GHK": ["BP13_GHK", "BP14_GHK"]}
    if good_sensor_map is None:
        good_sensor_map = {}

    out = {}
    # Hold empirical and model R(f) for optional plotting
    empirical_R = {}      # sensor -> (freqs_Hz_array, R_vals_array)
    model_R_funcs = {}    # sensor -> callable f_Hz -> R_model(f)

    for sensor, freq_map in data.items():
        # Skip bad sensors for now - process them at the end
        if sensor in bad_sensor_map:
            # Capture empirical points for plotting if needed
            f_emp = []
            R_emp = []
            for fk, m in freq_map.items():
                f_emp.append(float(fk.replace("k",""))*1e3 if fk.endswith("k") else float(fk))
                R_emp.append(m.R)
            empirical_R[sensor] = (np.array(f_emp), np.array(R_emp))
            continue

        # BPXX_ABK or BPXX_GHK: use average R/L and Z_R(w) correction
        if pat_abk.match(sensor) or pat_ghk.match(sensor):
            Rs = [m.R for m in freq_map.values()]
            Ls = [m.L for m in freq_map.values()]
            Cs = np.array([m.C for m in freq_map.values()])
            R_mean = np.mean(Rs) - R_0 # The wire resistance is added back in, in the skin depth correction
            L_mean = np.mean(Ls)
            def Z_R(w, R=R_mean, R_0=R_0):
                outZ = R_0 * np.ones_like(w)
                outZ[w/(2*np.pi)>=220e3] *= 1+(np.sqrt(w[w/(2*np.pi)>=220e3]/(2*np.pi))-np.sqrt(220e3))*0.0043
                outZ += R
                return outZ
            Z_L = lambda w, L=L_mean: (1j * w * L)
            Z_C = lambda w: 1 / (1j * w * (C_0 if Cs ==0 else Cs) )
            Z_total = lambda w: Z_R(w) + Z_L(w) + Z_C(w)
            H = lambda w: Z_C(w) / Z_total(w)
            mag_RLC = lambda w: np.abs(H(w))
            phase_RLC = lambda w: np.angle(H(w), deg=True)
            out[sensor] = {"mag_RLC": mag_RLC, "phase_RLC": phase_RLC}

            # Capture empirical points
            f_emp = []
            R_emp = []
            for fk, m in freq_map.items():
                f_emp.append(float(fk.replace("k",""))*1e3 if fk.endswith("k") else float(fk))
                R_emp.append(m.R)
            empirical_R[sensor] = (np.array(f_emp), np.array(R_emp))
            # Model R(f) as Z_R(w) with correction
            def _model_R_abkghk(f, Rm=R_mean, R0=R_0):
                w = 2*np.pi*np.asarray(f)
                outZ = R0 * np.ones_like(w)
                mask = (w/(2*np.pi) >= 220e3)
                if np.any(mask):
                    outZ[mask] *= 1 + (np.sqrt((w[mask]/(2*np.pi))) - np.sqrt(220e3)) * 0.0043
                outZ += Rm
                return outZ
            model_R_funcs[sensor] = _model_R_abkghk
            continue

        # For other sensors: fit R(f), L(f) if >1 point, else use mean
        fvals = []
        Rs = []
        Ls = []
        for fk, m in freq_map.items():
            if fk.endswith("k"):
                f = float(fk.replace("k", "")) * 1e3
            else:
                f = float(fk)
            fvals.append(f)
            Rs.append(m.R)
            Ls.append(m.L)
            Cs = np.array(m.C)
        fvals = np.array(fvals)
        Rs = np.array(Rs)
        Ls = np.array(Ls)
        Cs = np.array(Cs)
        if len(fvals) >= 2:
            # Fit R(f) and L(f) as linear functions of frequency
            R_coeffs = np.polyfit(fvals, Rs, 1)
            L_coeffs = np.polyfit(fvals, Ls, 1)
            def R_func(w):
                f = w/(2*np.pi)
                return np.polyval(R_coeffs, f)
            def L_func(w):
                f = w/(2*np.pi)
                return np.polyval(L_coeffs, f)
            # Also store model R(f) by frequency directly for plotting
            model_R_funcs[sensor] = lambda f, Rc=R_coeffs: np.polyval(Rc, np.asarray(f))
        else:
            R_mean = float(np.mean(Rs))
            L_mean = float(np.mean(Ls))
            R_func = lambda w: R_mean
            L_func = lambda w: L_mean
            model_R_funcs[sensor] = lambda f, Rm=R_mean: Rm

        def Z_R(w):
            return R_func(w)
        def Z_L(w):
            return 1j * w * L_func(w)
        Z_C = lambda w: 1 / (1j * w * (C_0 if Cs ==0 else Cs) )
        Z_total = lambda w: Z_R(w) + Z_L(w) + Z_C(w)
        H = lambda w: Z_C(w) / Z_total(w)
        mag_RLC = lambda w: np.abs(H(w))
        phase_RLC = lambda w: np.angle(H(w), deg=True)
        out[sensor] = {"mag_RLC": mag_RLC, "phase_RLC": phase_RLC}

        # Capture empirical points
        empirical_R[sensor] = (fvals, Rs)

    # Now process bad sensors: copy from first available good sensor
    for bad_sensor, good_list in bad_sensor_map.items():
        for good_sensor in good_list:
            if good_sensor in out:
                out[bad_sensor] = out[good_sensor]
                # Also copy model_R_funcs if it exists
                if good_sensor in model_R_funcs:
                    model_R_funcs[bad_sensor] = model_R_funcs[good_sensor]
                break

    # Optional test plot: empirical vs model R(f) for a few sensors
    if plot_R and empirical_R:
        # Choose a few sensors: up to 2 ABK, 2 GHK, 2 others, and up to 2 bad (if configured)
        abk_sensors = [s for s in data if re.match(r"^BP\d{2}_ABK$", s)]
        ghk_sensors = [s for s in data if re.match(r"^BP\d{2}_GHK$", s)]
        bad_sensors = [s for s in bad_sensor_map.keys() if s in data]
        others = [s for s in data if s not in abk_sensors and s not in ghk_sensors and s not in bad_sensors]

        ordered = []
        for bucket in (abk_sensors[:2], ghk_sensors[:2], others[:2], bad_sensors[:2]):
            for s in bucket:
                if s not in ordered:
                    ordered.append(s)
        pick = ordered[:6] if ordered else list(empirical_R.keys())[:6]

        f_model = np.logspace(3, 5, 200)  # 1 kHz to 100 kHz
        plt.figure(figsize=(6, 3.5),tight_layout=True)
        for ind,s in enumerate(pick):
            f_emp, R_emp = empirical_R.get(s, (None, None))
            if f_emp is None or s not in model_R_funcs:
                continue
            R_mod = model_R_funcs[s](f_model)
            # Model curve
            plt.plot(f_model*1e-3, R_mod, label=f"{s} model",c=plt.get_cmap('tab10')(ind))
            # Empirical points
            plt.plot(f_emp*1e-3, R_emp, 'o', ms=4, label=f"{s} empirical",c=plt.get_cmap('tab10')(ind))
        plt.xscale('log')
        plt.xlabel("Frequency [kHz]")
        plt.ylabel(r"$\Omega$ [Ohm]")
        plt.title("Empirical vs Model R(f) for selected sensors")
        plt.grid(True, which='both', ls=':')
        plt.legend(fontsize=8, ncol=2, handlelength=1)
        plt.tight_layout()
        plt.show()

    # Add in missing sensors by copying from specified good sensors
    for missing_sensor, good_sensor in missing_sensor_map.items():
        if good_sensor in out:
            out[missing_sensor] = out[good_sensor]
            if good_sensor in model_R_funcs:
                model_R_funcs[missing_sensor] = model_R_funcs[good_sensor]

    return out

def plot_RL_vs_frequency(data: Dict[str, Dict[str, RLMeasurement]], sensors: Optional[list[str]] = None):
    """
    Plot resistance (R) and inductance (L) vs frequency for each sensor.
    If sensors is provided, only plot those sensors.
    """

    freq_keys = ["1k", "10k", "100k"]
    freqs = [FREQUENCIES_HZ[k] for k in freq_keys]

    if sensors is None:
        sensors = sorted(data.keys())

    for name in sensors:
        freq_present = []
        R_vals = []
        L_vals = []
        for fk, f in zip(freq_keys, freqs):
            meas = data.get(name, {}).get(fk)
            if meas is not None:
                freq_present.append(f)
                R_vals.append(meas.R)
                L_vals.append(meas.L)
        if not freq_present:
            continue
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(freq_present, R_vals, marker='o')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Resistance (Ohm)')
        plt.title(f'{name} - Resistance')
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(freq_present, L_vals, marker='o')
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Inductance (H)')
        plt.title(f'{name} - Inductance')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # # Lightweight CLI to clean the calibration spreadsheet.
    # import argparse

    # parser = argparse.ArgumentParser(description="Clean and average C-Mod R/L calibration CSV.")
    # parser.add_argument("csv", nargs="?", default="C_Mod_Calibrated_RLC.csv",
    #                     help="Path to input CSV (default: C_Mod_Calibrated_RLC.csv)")
    # parser.add_argument("--out", default="C_Mod_Calibrated_RLC_cleaned.csv",
    #                     help="Output CSV path (default: C_Mod_Calibrated_RLC_cleaned.csv)")
    # parser.add_argument("--R-min", type=float, default=DEFAULT_R_MIN, dest="R_min",
    #                     help=f"Minimum R (Ohm), default {DEFAULT_R_MIN}")
    # parser.add_argument("--R-max", type=float, default=DEFAULT_R_MAX, dest="R_max",
    #                     help=f"Maximum R (Ohm), default {DEFAULT_R_MAX}")
    # parser.add_argument("--L-min", type=float, default=DEFAULT_L_MIN, dest="L_min",
    #                     help=f"Minimum L (H), default {DEFAULT_L_MIN}")
    # parser.add_argument("--L-max", type=float, default=DEFAULT_L_MAX, dest="L_max",
    #                     help=f"Maximum L (H), default {DEFAULT_L_MAX}")
    # parser.add_argument("--L-scale", type=float, default=1e-6, dest="L_scale",
    #                     help="Scale to apply to L values before thresholding (default: 1e-6 for microhenry→henry)")
    # parser.add_argument("--summary", action="store_true", help="Print a short summary to stdout")

    # args = parser.parse_args()

    # data = load_calibrated_RL_csv(
    #     args.csv, R_min=args.R_min, R_max=args.R_max, L_min=args.L_min, L_max=args.L_max, L_scale=args.L_scale
    # )
    # write_cleaned_RL_csv(data, args.out)

    # if args.summary:
    #     total_pairs = sum(len(freqs) for freqs in data.values())
    #     sensors = len(data)
    #     print(f"Wrote {args.out}: {sensors} sensors, {total_pairs} (sensor,freq) pairs")
    #     # Show a few example lines
    #     shown = 0
    #     for name, freqs in data.items():
    #         for fk, m in freqs.items():
    #             print(f"  {name} @ {fk}: R={m.R:.3g} Ω, L={m.L:.3g} H")
    #             shown += 1
    #             if shown >= 3:
    #                 break
    #         if shown >= 3:
    #             break

    # # Plot all sensors from the cleaned CSV
    # try:
    #     cleaned_data = load_calibrated_RL_csv("C_Mod_Calibrated_RLC_cleaned.csv")
    #     plot_RL_vs_frequency(cleaned_data)
    # except Exception as e:
    #     print(f"Could not plot cleaned CSV: {e}")


    __prep_RLC_Transfer_Function(plot_R=True)


    print("Done.")
