# Categorical Decision Tree

Implementation of a Decision Tree for categorical features.
The algorithm is based on the CART Algorithm.
> Breiman, L. (2017). _Classification and regression trees_. Routledge.

This implementation allows to build a Decision Tree from categorical features, being able to modify parameters like maximum tree depth, or the threshold to predict a sample.


## Example
An example of how to use CatDTC is provided in the Jupyter Notebook **Example.ipynb**

## Future Work

- **Add Gini Criterion**: Right now entropy is only implemented as a split criterion when building the tree.
- **Allow Multiclass Target Variable**: This implementation just allows binary target variables.
- **Evaluation Model**: Create a evaluation model with different metrics such as, accuracy, AUC or RMSE.
