from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_default_preprocessing():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=-1),
             make_column_selector(dtype_exclude="category")),
            ("cat", Pipeline(steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=-1))
                                    ]),
             make_column_selector(dtype_include="category")),
        ],
        sparse_threshold=0
    )

    return preprocessor
