#import imbalanced_databases as imbd
from collections import Counter
import numpy as np

def get_data(name, val_ratio=0.2, test_raito=0.2, undersample_ratio=100):
    from sklearn.datasets import fetch_openml
    from imblearn.datasets import make_imbalance
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.compose import make_column_selector as selector
    from sklearn import preprocessing
    version=1
    X, y = fetch_openml(name, version=version, as_frame=True, return_X_y=True)
    num_pipe = make_pipeline(
        StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
    )
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    preprocessor_linear = make_column_transformer(
        (num_pipe, selector(dtype_include="number")),
        (cat_pipe, selector(dtype_include="category")),
        n_jobs=2,
    )
    X = preprocessor_linear.fit_transform(X)

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    if type(X) != np.ndarray:
        X = X.todense()

    print("========== Dataset Info ==========")
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Counts:", Counter(y))
    print("==================================")
    size = X.shape[0]
    # Shuffle
    indices = [i for i in range(size)]
    np.random.shuffle(indices)
    val_idx, test_idx, train_idx = indices[:int(size*val_ratio)], indices[int(size*val_ratio):int(size*(val_ratio+test_raito))], indices[int(size*(val_ratio+test_raito)):]

    train_X, val_X, test_X = X[train_idx], X[val_idx], X[test_idx]
    train_y, val_y, test_y = y[train_idx], y[val_idx], y[test_idx]

    class_counts = Counter(train_y)
    train_X, train_y = make_imbalance(
        train_X,
        train_y,
        sampling_strategy={
            min(class_counts, key=class_counts.get): max(class_counts.values()) // undersample_ratio,
        },
    )
    class_counts = Counter(val_y)
    val_X, val_y = make_imbalance(
        val_X,
        val_y,
        sampling_strategy={
            min(class_counts, key=class_counts.get): max(class_counts.values()) // undersample_ratio,
        },
    )
    class_counts = Counter(test_y)
    test_X, test_y = make_imbalance(
        test_X,
        test_y,
        sampling_strategy={
            min(class_counts, key=class_counts.get): max(class_counts.values()) // undersample_ratio,
        },
    )

    print("========== Training Info =========")
    print("train X shape:", train_X.shape, "train y shape:", train_y.shape)
    print("Counts:", Counter(train_y))
    print("==================================")

    return train_X, train_y, val_X, val_y, test_X, test_y

