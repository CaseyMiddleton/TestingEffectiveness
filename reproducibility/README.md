The manuscript that this repository supports contains a number of results and figures. This `reproducibility` folder is meant to facilitate the complete reproduction of those results and figures.

# Reproducing Results and Figures
1. Copy all files contained in `reproducibility/input` into the `input` folder of the base directory. 
2. Copy the two scripts `run_all_main.sh` and `run_all_supp.sh` into the base directory.
3. Run both shell scripts by executing from the command line:
```
sh run_all_main.sh
sh run_all_supp.sh
```
4. Uncomment the desired figures in the `reproducibility/figures.py` main() function and run script from the commmand line.

# Reproducing Area-Under-The-Curve Plots
1. Open `visualize_a_simulation.ipynb` (which uses `viz_utils.py`) and run it from this directory. 
2. See notes within the Jupyter notebook for customizations including: parameters, number of samples, color, size, axes, and more.
