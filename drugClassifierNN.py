# Filename: drugClassifierNN
# Dependencies: drugClassifierNNUtilities, sklearn, numpy
# Author: Jean-Michel Boudreau
# Date: April 29, 2019

'''
The purpose of this script is to determine how sufficient Lipinsky's Rule of 5 
- a heuristic set of 4 rules that allow one to determine whether a drug will be 
bioactive or not - is in determining whether a drug should be classified as a 
drug or not. This is done by comparing a DNN trained with 4 features 
corresponding to drugs that either meet or don't meet Lipinsky's Rule of 5 with
another DNN trained with an additional 5 features (9 total) that correspond to
improvements made to the Lipinsky model. The improvements to the Lipinsky model
are taken from the conditions listed on the ZINC db for "drug-like" molecules:

1) 150 Dalton <=  Molecular Weight <= 500 Dalton
2) Xlog(P) <= 5 
3) Number of Rotatable Bonds <=7 
4) Polar Surface Area < 150 Angstroms (Extension of Veber Rules)
5) No. H Bond Donors <= 5
6) No. H Bond Acceptors <= 10

The test set in each case was molecules identified as "drug-like" according to 
the ZINC db.

A short summary of the methodology is given below:
The entire ZINC db catalog of molecules (>22 mil. entries) is used to create 4 
libraries of molecules: two distinct libraries that contains "drug-like" 
molecules (may or may not contain unique entries between the two libraries) and
another two that contain "nondrug-like" molecules. One library of each kind 
is used to train two distinct deep neural networks, one in which the features 
of the training subset is restricted to four of the numerical attributes
(corresponding to the well-known Lipinsky's rules) and another in which all 
numerical features are used to train the DNN. All features were ca. by 
“Molinspiration MiTools” molecular processing toolkit.

Uncomment line 47 to download the ZINC db catalog into the same folder
Uncomment line 54 to inspect a few histograms of the numerical data
'''

# Import Libraries
from drugClassifierNNUtilities import fetch_zinc_data, load_zinc_data, inspect_data, Lipinsky_drug_classifier_v1, Lipinsky_drug_classifier_v2, process_data, DNN
from sklearn.metrics import accuracy_score
import numpy as np

# Download the data into current working directory. Saved as "zinc_db.xls"
# fetch_zinc_data()


# Load the ZINC db as pandas dataframe
zinc_data = load_zinc_data()

# Inspect data
# inspect_data(zinc_data)

# Lipinsky's Rule of 5 (strict and modified, lip1 and lip2 resp.) data subsets
lip1_drug_data, lip1_non_drug_data = Lipinsky_drug_classifier_v1(zinc_data)
lip2_drug_data, lip2_non_drug_data = Lipinsky_drug_classifier_v2(zinc_data)

# Free memory being used for zinc_data
del zinc_data

# Process the data and feed into neural network one at a time
n_mol = 100000 # total number of molecules to be processed
# Lipisnky Rule of 5 (without additional features) data subset:
# Prepare data for DNN
version = 1
X_train_v1, X_valid_v1, y_train_v1, y_valid_v1 = process_data(
        lip1_drug_data, lip1_non_drug_data, n_mol, version)

# Lipisnky Rule of 5 (with additional features) data subset
# Prepare data for DNN
version = 2
X_train_v2, X_valid_v2, X_test_v2, y_train_v2, y_valid_v2, y_test = process_data(
        lip2_drug_data, lip2_non_drug_data, n_mol, version)

# To test both models on same set of molecules 
X_test_v1 = np.delete(X_test_v2, [2,3,6,7,8], 1)

# Train and test neural network with data and store results
y_pred_lip1 = DNN(4, 7, 5, 2, version, 
                  X_train_v1, X_valid_v1, X_test_v1, y_train_v1, y_valid_v1, y_test)
# Performance measure for comparing the models
accuracy_lip1 = accuracy_score(y_test, y_pred_lip1)

# Train and test neural network with data and store results
y_pred_lip2 = DNN(9, 7, 5, 2, version, 
                  X_train_v2, X_valid_v2, X_test_v2, y_train_v2, y_valid_v2, y_test)
# Performance measure for comparing the models
accuracy_lip2 = accuracy_score(y_test, y_pred_lip2)