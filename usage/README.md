# Usage of the Package

The `tutorial.py` contains an example usage of the package. Not only two training loops are shown, but also the filtering of the permutations and some sample data is provided for illustration.

## Training loops

Only two lines of code are needed for the training loop. The most relevant options in both functions are already presented and for further details the [documentation](https://search-for-eps.github.io/Search-for-EPs/) provides more information. The last three line in the training loop are for the evaluation of the obtained results. A plot is saved as a `html` in the current directory which shows the predictioons of the exceptional point in each training step. The corresponding data is saved in a `csv` file. The kernel eigenvalues are saved in a `pickle` file if further evaluation is required.
The second training loop shows which arrays need to be sliced if less data points want to be used for training the model.

## Initial training set

To obtain an initial training set, the permutations have to be filtered. This is done in this function and the permutations found are plotted. After that, the first commented out line can be used to identify the columns belonging to one permutation. The last commented out line is then used to save the data of the selected permutation to a `csv` file. The `get_permutations` function has an optional argument `distance` which can be varried if necessary, usually depending on the radius of the ellipse.
