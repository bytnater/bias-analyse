# bias-analyse
No single bias metric provides a complete picture of “fairness”; you will almost always need to report several and understand the trade-offs. We explore whether a structured approach can be developed to calculate bias and fairness metrics given the dataset.

# Project Overview
This project contains the source code for a structured approarch to compute bias and fairness metrics given a dataset. we also looked supplementary items to better see which features may have bias. Such as [SHAP](https://shap.readthedocs.io/en/latest/) and Hierarchical Bias-Aware Clustering (HBAC) according to our interpretation of the paper [Auditing a dutch public sector risk profiling algorithm using an unsupervised bias detection tool](https://arxiv.org/pdf/2502.01713).

## Technologies
- Python
- Jupyter Notebook

## Notebooks
In [main.ipynb](https://github.com/bytnater/bias-analyse/blob/main/main.ipynb) you can select and analyze any .csv file using different fairness metrics.
1. Run the first cell and upload the .csv file to analyze.
2. Select which features are the ground truth, predictions, and/or which are to be analyzed.
3. Select which metrics to use
4. Compute and visualize the results
5. (optional) Save selections for later use. 

In [SHAP.ipynb](https://github.com/bytnater/bias-analyse/blob/main/SHAP.ipynb) is code where you can upload a model to analyze for feature importance. This can be usefull to gain insight which of the protected features may be prone to bias. 

In [HBAC.ipynb](https://github.com/bytnater/bias-analyse/blob/main/HBAC.ipynb) is the code for use our implementation of Hierarchical Bias-Aware Clustering. This method tries to uncover subpopulations where there is high group bias.

In [synth_data.ipynb](https://github.com/bytnater/bias-analyse/blob/main/data/synth_data.ipynb) is the code for generating test data to use for our fairness metrics.

