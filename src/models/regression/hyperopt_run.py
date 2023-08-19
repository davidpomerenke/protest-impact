from src.models.regression.hyperopt import hyperopt

if __name__ == "__main__":
    for model in ["ols", "ridge", "lasso"]:
        print(model)
        best_result = hyperopt(model, n_jobs=4)
        print(best_result)
