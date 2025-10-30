# 🎯 Supervised Machine Learning 
**Definition**: Supervised Machine Learning (ML) is a type of ML where the model is trained on a labeled dataset—each input comes with the correct output. The model learns a mapping from inputs to outputs so it can predict the output for new, unseen inputs.

**Key points**:
- Requires labeled data $(X, Y)$.
- Goal: Learn a function $f(X) \to Y$.

Depending on the type of target $Y$, the supervised ML can be categorized as one of two problem:
- **Regression**: predicting house prices, stock forecasting.
- **Classfication**: spam detection, image recognition. 

### ✅ Important point
- Searching in function space may lead to infinite solutions. So, we need to make some assumption on the nature of function we are going to find in order for us to solve the problem. 
  - The assumption about the underlying problem introduces biases, so-called **inductive bias**. 
- With this in mind, ML algorithms are different in the form of assumption applying to the problem. 

## Contents 
- [Linear Regression](./README_LINEAR_REGRESSION.md)
- [Probabilistic Classifiers](./README_BAYES_CLASSIFIERS.md)
- [Linear Classifiers](./README_LINEAR_CLASSIFIERS.md)
- [Support Vector Machines](./README_SVM.md)
- [Neural Networks](./README_NEURAL_NETWORKS.md)
- [Decision Trees](./README_DECISION_TREE.md)
- [Ensemble Models](./README_ENSEMBLE_MODELS.md)
