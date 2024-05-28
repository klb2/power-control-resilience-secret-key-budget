# Template for Reproducible Research Papers

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/power-control-resilience-secret-key-budget)
![GitHub](https://img.shields.io/github/license/klb2/power-control-resilience-secret-key-budget)


This repository is accompanying the paper "Power Control for Resilient
Communication Systems With a Secret-Key Budget" (Karl-L. Besser, Rafael
Schaefer, and Vincent Poor, IEEE International Symposium on Personal, Indoor
and Mobile Radio Communications (PIMRC), Sep. 2024).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.


## File List
The following files are provided in this repository:

- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `util.py`: Python module that contains utility functions, e.g., for saving results.
- `constant_power.py`: Python script that runs the main simulation with a
  constant power allocation.
- `ide_stopping_time.py`: Python module that contains functions to calculate
  the stopping times numerically via solving an IDE.
- `mc_stopping_time.py`: Python module that contains functions to determine the
  stopping times via Monte Carlo simulations.
- `ultimate_ruin_prob.py`: Python module that contains functions to calculate
  the probability of ultimate ruin (for the random timing scheme).
- `rayleigh.py`: Python module that contains functions related to Rayleigh
  fading (distribution functions, ...)


## Usage
### Running it online
You can use services like [CodeOcean](https://codeocean.com) or
[Binder](https://mybinder.org/v2/gh/klb2/power-control-resilience-secret-key-budget/HEAD)
to run the scripts online.

### Local Installation
If you want to run it locally on your machine, Python3 and Jupyter are needed.
The present code was developed and tested with the following versions:

- Python 3.12
- numpy 1.26
- scipy 1.13
- matplotlib 3.9
- pandas 2.2

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file. 
You can then recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported by the German Research Foundation (DFG) under
grants BE 8098/1-1 and SCHA 1944/11-1, by the German Federal Ministry of
Education and Research (BMBF) within the national initiative on 6G
Communication Systems through the research hub 6G-life under Grant 16KISK001K
as well as the 6G-ANNA project under Grant 16KISK103, and by the U.S. National
Science Foundation under Grants CNS-2128448 and ECCS-2335876.


## License and Referencing
This program is licensed under the MIT license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@inproceedings{Besser2024pimrc,
  author = {Besser, Karl-Ludwig and Schaefer, Rafael F. and Poor, H. Vincent},
  title = {Power Control for Resilient Communication Systems With a Secret-Key Budget},
  booktitle = {2024 IEEE International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)},
  year = {2024},
  month = {9},
  publisher = {IEEE},
  venue = {Valencia, Spain},
}
```
