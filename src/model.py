# Trains and evaluates the spread models. The whole file is built around one
# rule: a model is only ever scored on seasons it was never trained on. Validation
# is walk-forward by season (train on the past, test on the next year) rather than
# a random split -- a random k-fold would mix future games into training and make
# everything look better than it actually is.
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor


# For each fold, train on every season up to year i
# and test on year i+1
def walk_forward_by_season(dfs_by_year, fit_predict_fn, target_col = "margin",
        feature_cols = None, dropna = True,):
    years = sorted(dfs_by_year.keys())
    rows = []

    for i in range(len(years) - 1):
        train_years = years[:i + 1]
        test_year = years[i + 1]

        train_df = pd.concat([dfs_by_year[y] for y in train_years], ignore_index=True)
        test_df = dfs_by_year[test_year].copy()
        # default to every numeric column except the target
        if feature_cols is None:
            num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            feat_cols = [c for c in num_cols if c != target_col]
        else:
            feat_cols = feature_cols

        train_df[feat_cols] = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
        test_df[feat_cols] = test_df[feat_cols].apply(pd.to_numeric, errors="coerce")

        if dropna:
            train_df = train_df.dropna(subset=[target_col] + feat_cols)
            test_df = test_df.dropna(subset=[target_col] + feat_cols)

        X_train, y_train = train_df[feat_cols], train_df[target_col]
        X_test, y_test = test_df[feat_cols], test_df[target_col]

        y_pred_test = fit_predict_fn(X_train, y_train, X_test)
        y_pred_train = fit_predict_fn(X_train, y_train, X_train)

        rows.append({
            "fold": f"{train_years[0]}-{train_years[-1]} -> {test_year}",
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_MAE": mean_absolute_error(y_train, y_pred_train),
            "test_MAE": mean_absolute_error(y_test, y_pred_test),
            "train_RMSE": root_mean_squared_error(y_train, y_pred_train),
            "test_RMSE": root_mean_squared_error(y_test, y_pred_test),
        })

    return pd.DataFrame(rows)

# Model definitions
# Each of these returns a small fit-predict closure shaped like
# (X_train, y_train, X_test) -> predictions. Giving every model the same shape is
# what lets walk_forward_by_season swap them in and out without knowing or caring
# which estimator it's running. Hyperparameters are baked in here.

# Ridge regressor
def get_ridge_fn(alpha = 1.0):
    return lambda X_train, y_train, X_test: (
        Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)
    )

# ElasticNet
def get_elasticnet_fn(alpha = 0.001, l1_ratio = 0.5):
    return lambda X_train, y_train, X_test: (
        Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=0))
        ])
        .fit(X_train, y_train)
        .predict(X_test)
    )

# Histogram gradient boosting
def get_hgb_fn():
    return lambda X_train, y_train, X_test: (
        HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.01,
            max_depth=4,
            max_leaf_nodes=63,
            min_samples_leaf=30,
            l2_regularization=0.3,
            random_state=0,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=80,
            max_features=0.7,
            validation_fraction=0.1,
        )
        .fit(X_train, y_train)
        .predict(X_test)
    )

# random forest
def get_rf_fn():
    return lambda X_train, y_train, X_test: (
        RandomForestRegressor(
            n_estimators=1200,
            max_depth=10,
            min_samples_leaf=50,
            min_samples_split=100,
            max_features=0.4,
            bootstrap=True,
            n_jobs=-1,
            random_state=0
        )
        .fit(X_train, y_train)
        .predict(X_test)
    )

# XGBoost
def get_xgb_fn():
    return lambda X_train, y_train, X_test: (
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=12000,
            learning_rate=0.0006,
            max_depth=3,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_lambda=10.0,
            reg_alpha=0.1,
            gamma=0.5,
            random_state=0,
            n_jobs=-1,
        )
        .fit(X_train, y_train)
        .predict(X_test)
    )

# Huber regressor - used as the final model for predictions
# Huber loss is robust to outliers which matters because college basketball is prone to blowouts
# fit intercept is off since features are already centered toward 0 because they are differentials
def get_huber_fn(epsilon = 3, alpha = 0.001):
    return lambda X_train, y_train, X_test: (
        Pipeline([
            ("huber", HuberRegressor(
                epsilon=epsilon,
                alpha=alpha,
                max_iter=30000,
                fit_intercept=False,
            ))
        ])
        .fit(X_train, y_train)
        .predict(X_test)
    )

# Small neural network with early stopping. To test if nonlinearity would improve
# model performance on the feature set
def get_mlp_fn():
    return lambda X_train, y_train, X_test: (
        Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-2,
                learning_rate_init=1e-5,
                max_iter=25000,
                batch_size=64,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=150,
                random_state=0
            ))
        ])
        .fit(X_train, y_train)
        .predict(X_test)
    )


# Refits huber on all seasons with no holdout
def train_final_model(dfs_by_year, target_col = "margin", feature_cols = None, dropna = True,
        model_path = "huber_margin_model.pkl", features_path = "feature_cols_huber.pkl",):
    all_df = pd.concat(dfs_by_year.values(), ignore_index=True)

    if feature_cols is None:
        num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c != target_col]
    else:
        feat_cols = feature_cols

    all_df[feat_cols] = all_df[feat_cols].apply(pd.to_numeric, errors="coerce")

    if dropna:
        all_df = all_df.dropna(subset=[target_col] + feat_cols)

    X = all_df[feat_cols]
    y = all_df[target_col]

    model = HuberRegressor(
        epsilon=3,
        alpha=0.001,
        max_iter=30000,
        fit_intercept=False
    )
    model.fit(X, y)

    # Calculate feature importance by measuring how much MAE gets worse from shuffling features
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=0,
        scoring='neg_mean_absolute_error'
    )

    importance_df = pd.DataFrame({
        'feature': feat_cols,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    # Save the model and feature list
    joblib.dump(model, model_path)
    joblib.dump(feat_cols, features_path)

    print(f"Model saved to {model_path}")
    print(f"Features saved to {features_path}")

    return model, importance_df, feat_cols

# Reload a saved model and its feature list
def load_model(model_path = "huber_margin_model.pkl", features_path = "feature_cols_huber.pkl"):
    model = joblib.load(model_path)
    feat_cols = joblib.load(features_path)
    return model, feat_cols


# Compare the performance of all models trained and store in table
def compare_models(dfs_by_year, target_col = "margin"):
    models = {
        "Ridge": get_ridge_fn(),
        "ElasticNet": get_elasticnet_fn(),
        "HGB": get_hgb_fn(),
        "RandomForest": get_rf_fn(),
        "XGBoost": get_xgb_fn(),
        "Huber": get_huber_fn(),
        "MLP": get_mlp_fn(),
    }

    results = []
    for name, fn in models.items():
        print(f"Evaluating {name}...")
        result = walk_forward_by_season(dfs_by_year, fn, target_col=target_col)
        results.append({
            "model": name,
            "avg_test_MAE": result["test_MAE"].mean(),
            "avg_test_RMSE": result["test_RMSE"].mean(),
            "avg_train_MAE": result["train_MAE"].mean(),
            "overfit_gap": result["train_MAE"].mean() - result["test_MAE"].mean(),
        })

    return pd.DataFrame(results).sort_values("avg_test_MAE")


# Script entry point
if __name__ == "__main__":
    from historical_data import load_historical_data

    print("Loading historical data...")
    merged_dfs, model_dfs = load_historical_data()

    print("\nComparing models...")
    comparison = compare_models(model_dfs)
    print("\n" + comparison.to_string(index=False))

    print("\nTraining final Huber model...")
    model, importance_df, feat_cols = train_final_model(model_dfs)
    print("\nFeature importance:")
    print(importance_df)