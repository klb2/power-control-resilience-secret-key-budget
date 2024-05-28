import os
import re
import argparse

import pandas as pd

from constant_power import RESULTS_DIR

PARAMETERS = {"prob_tx": ("p", float),
              "power_tx_db": ("P", float),
              "init_budget": ("b", float),
              "target_alert_outage_prob": ("e", float),
              }

def _create_filename_parser():
    parser = argparse.ArgumentParser(add_help=False)
    for _param, (_abbr, _type) in PARAMETERS.items():
        parser.add_argument(f"-{_abbr}", f"--{_param}", type=_type)
    return parser

def main(times, results_dir=RESULTS_DIR):
    new_results = pd.DataFrame()
    data_files = [f for f in os.listdir(results_dir) if f.endswith(".dat")]
    filename_parser = _create_filename_parser()
    for _file in data_files:
        _filename = os.path.splitext(_file)[0]
        _parts = re.split(r'(-[a-zA-Z])', _filename)
        if not _parts[0] == "results": continue
        params = filename_parser.parse_args(_parts[1:])
        params = vars(params)
        
        _data = pd.read_csv(os.path.join(results_dir, _file), sep='\t')
                            #index_col="time")
        for _param, _value in params.items():
            _data[_param] = _value
        new_results = pd.concat((new_results, _data.loc[times]))

    grouping_params = ["time", "prob_tx", "init_budget", "target_alert_outage_prob"]
    for _group_params, _combined_results in new_results.groupby(grouping_params):
        _combined_results = _combined_results.sort_values("power_tx_db")
        _filename_parts = ["{0}{1:.3f}".format(PARAMETERS.get(_p, ["t"])[0], _v)
                           for _p, _v in zip(grouping_params, _group_params)]
        filename = "combined_results-{}.dat".format('-'.join(_filename_parts))
        filename = os.path.join(results_dir, filename)
        _combined_results.to_csv(filename, sep='\t', index=False)
    return new_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("-t", "--times", nargs="+", required=True, type=int)
    args = vars(parser.parse_args())
    main(**args)
