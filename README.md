# Prediction of Breast Cancer Using Supervised Machine Learning Techniques
We obtained the breast cancer dataset from UCI repository and used Jupyter Notebook as the platform for the purpose of coding. Our methodology involves use of classification techniques like Support Vector Machine (SVM) and K-Nearest Neighbor (K-NN).
## A. Feature Selection
Feature selection is finding the subset of original features by different approaches based on the information they provide, accuracy, prediction errors.
The features used in the project are:
- Clump_thickness 
- Uniform_cell_size 
- Uniform_cell_shape 
- Marginal_adhesion 
- Single_epithelial_size 
- Bare_nuclei 
- Bland_chromatin 
- Normal_nucleoli 
- Mitoses
## B. Model Selection
* SVC/SVM model with Linear kernel
* KNN model with nearest neighbours set eqaul to 5
## C. Training the models with Data
The data taken is from **https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data**
## D. Taining Data and Testing Data
80% of above data is training and 20% is testing data.
### Then the Class is predicted:
- 2 if Benign Cell
- 4 if Malignant Cell
# Result =>
Both algorithms have almost same accuracy(mostly greater than 90%)

## Files included in repository are:
- **source.ipynb(Jupyter Notebook-https://jupyter.org/)**
- **source.pdf(Just a pdf print of jupyter notebook)**
- **breast-cancer-wisconsin.data(File that stores Train and Test Data)**  <br />

***In source.ipynb, data is visualized using Histograms and Scatter Plots.***
