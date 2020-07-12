## Additional Parameters 

In this file, we would like to note some parameters used in `preprocess_data.py` and `generate_loaders.py`. 
Feel free to modify it according to your needs.

---

**Parameters in `preprocess_data.py`**
* Line 81, 179: Modify three parameters, topk_ratio, adj, option, for choosing risky profiles.
* Line 148-150: Modify the length of train, validation, and test data
* Line 175: Select the features for risky profiles
* Line 185: Decide which columns to save

---

**Parameters in `generate_loaders.py`**
* Line 45: Decide which columns to use for XGB model
* Line 49: XGBClassifier, please refer [API](https://xgboost.readthedocs.io/en/latest/python/python_api.html) for its parameters
* Line 62, 101: Choose relevant criteria to observe the performance
* Line 90: LogisticRegression, please refer [API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for its parameters


