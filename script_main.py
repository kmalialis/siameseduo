# -*- coding: utf-8 -*-
from main import main

lst_ds = [
    # The following will create in the folder "exps" the results for SiameseDuo++ corresponding to Fig. 8b  in the paper
    {'repeats': 2, 'data_source': 'sea_abrupt',
     'memory': 10, 'method': 'actisiamese', 'flag_learning': 'active', 'active_budget_total': 0.01,
     'flag_da': True, 'da_method': ['interpolation', 'extrapolation', 'gaussian_noise'], 'da_n_generated': 5,
     'da_beta': 0.1}
]

if __name__ == "__main__":
    for d in lst_ds:
        main(d)
