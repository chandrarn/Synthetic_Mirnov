# Test of using disruption-py to generate spectrograms for C-Mod Mirnov data
from sys import path
path.append('/home/rianc/Documents/disruption-py/')
import disruption_py as dp
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data

import numpy as np
import matplotlib.pyplot as plt
from disruption_py.machine.tokamak import Tokamak, resolve_tokamak_from_environment
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.machine.cmod import mirnov


tokamak = resolve_tokamak_from_environment()

run_methods = ["get_all_mirnov_ffts"]
shotlist = [1110316031, 1160714026]

retrieval_settings = RetrievalSettings(
    run_methods=run_methods,
    efit_nickname_setting="default",
    time_setting="mirnov",
)

result = get_shots_data(
    tokamak=tokamak,
    shotlist_setting=shotlist,
    retrieval_settings=retrieval_settings,
    output_setting="dataset",
)


print(result)
