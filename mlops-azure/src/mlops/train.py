import os
import joblib
from azureml.core import Run
import argparse
import pandas as pd
from azureml.core import Workspace, Datastore, Dataset
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


__here__ = os.path.dirname(__file__)


# Custom Transformer to create the 'has_credit_account' feature
class CreateCreditAccountFeature(BaseEstimator, TransformerMixin):
    def __init__(self, no_credit_account_value):
        self.no_credit_account_value = no_credit_account_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['has_credit_account'] = X_copy['credit_account_id'].apply(lambda x: 0 if x == self.no_credit_account_value else 1)
        return X_copy


def train_model(X_train):
    # Define the pipeline
    pipeline = Pipeline([
        ('create_credit_account_feature', CreateCreditAccountFeature(no_credit_account_value=
                                                                     '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=5, random_state=42))
    ])

    pipeline.fit(X_train)
    return pipeline

def save_model(classifer):
    output_dir = os.path.join(__here__, 'models')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(classifer, model_path)
    return model_path

def register_model(run, model_path):
    run.upload_file(model_path, "models/model.pkl")
    model = run.register_model(
        model_name='cust_segm_clusterer',
        model_path="models/model.pkl"
    )
    run.log('Model_ID', model.id)


def main():
    parser = argparse.ArgumentParser("parser")
    parser.add_argument("--arg1", type=str)
    args = parser.parse_args()
    run = Run.get_context()
    workspace = run.experiment.workspace

    datastore = Datastore.get(workspace, "workspaceblobstore")
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'UI/2024-02-13_160843_UTC/customer_data_sample.csv'))
    df = dataset.to_pandas_dataframe()

    # train model
    model = train_model(df)

    model_path = save_model(model)
    register_model(run, model_path)




if __name__ == '__main__':
    main()
