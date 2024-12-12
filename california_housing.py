from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
from xgboost import XGBRegressor
import joblib
from scipy import stats

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing_train = strat_train_set.copy()
housing_test = strat_test_set
housing_train_label = housing_train["median_house_value"].copy()
housing_train = housing_train.drop("median_house_value", axis=1)
housing_test_label = housing_test["median_house_value"].copy()
housing_test = housing_test.drop("median_house_value", axis=1)

imputer = SimpleImputer(strategy="median")
housing_num = housing_train.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_train["median_house_value"] = housing_train_label

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)
housing_train = housing_train.iloc[outlier_pred == 1]
housing_train["bedroom_room_ratio"] = housing_train["total_bedrooms"]/housing_train["total_rooms"]
housing_train["people_per_house"] = housing_train["population"]/housing_train["households"]
housing_train["rooms_per_house"] = housing_train["total_rooms"]/housing_train["households"]
housing_train["people_per_room"] = housing_train["population"]/housing_train["total_rooms"]
housing_train = housing_train[~((housing_train["population"] < 300) | (housing_train["people_per_house"] > 6) | (housing_train["rooms_per_house"] > 10))]
housing_train.drop(columns=["bedroom_room_ratio", "people_per_house", "rooms_per_house", "people_per_room"], inplace=True)

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))
    
set_config(display='diagram')

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=60, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing = ColumnTransformer(
    transformers=[
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                  "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),

        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline
)

param = {'colsample_bytree': 0.23156174575739172,
 'learning_rate': 0.03355303228675172,
 'max_depth': 10,
 'min_child_weight': 2,
 'n_estimators': 465,
 'reg_alpha': 0.5479953975561015,
 'reg_lambda': 0.45507726952370925,
 'subsample': 0.9744976572285455}

xgb_reg = Pipeline([("preprocessing", preprocessing),
                    ("xgb", XGBRegressor(random_state=42))])

xgb_reg.named_steps["xgb"].set_params(**param)
housing_train_label = housing_train['median_house_value']
housing_train.drop("median_house_value", axis=1, inplace=True)
xgb_rmses = -cross_val_score(xgb_reg, housing_train, housing_train_label,
                                scoring="neg_root_mean_squared_error", cv=3)
print(pd.Series(xgb_rmses))

housing_train["median_housing_price"] = housing_train_label
strat_train_set, strat_test_set = train_test_split(
 housing_train, test_size=0.20, random_state=42)
strat_train_labels = strat_train_set["median_housing_price"]
strat_train_housing = strat_train_set.drop("median_housing_price", axis=1)
strat_test_labels = strat_test_set["median_housing_price"]
strat_test_housing = strat_test_set.drop("median_housing_price", axis=1)
xgb_reg.fit(strat_train_housing, strat_train_labels)
housing_predictions = xgb_reg.predict(strat_test_housing)
xgb_rmse = root_mean_squared_error(strat_test_labels, housing_predictions,)

print(xgb_rmse)
housing_train.drop("median_housing_price", axis=1, inplace=True)
xgb_reg.fit(housing_train, housing_train_label)
housing_predictions = xgb_reg.predict(housing_test)
xgb_rmse = root_mean_squared_error(housing_test_label, housing_predictions)

print(xgb_rmse)

confidence = 0.95
squared_errors = (housing_predictions - housing_test_label) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
joblib.dump(xgb_reg, "my_california_housing_model.pkl")