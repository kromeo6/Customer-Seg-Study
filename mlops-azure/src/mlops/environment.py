import os
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


ENVIRONMENT = 'env'

def create_aml_environment():
    aml_env = Environment(name=ENVIRONMENT)
    conda_dep = CondaDependencies()
    conda_dep.add_pip_package("pandas==1.5.3")
    conda_dep.add_pip_package("joblib==1.3.2")
    conda_dep.add_pip_package("scikit-learn==1.2.0")
    conda_dep.add_pip_package("azureml-mlflow==1.38.0")
    aml_env.python.conda_dependencies = conda_dep
    aml_env.docker.enabled = True
    return aml_env


def main():
    workspace_name = os.environ['AML_WORKSPACE_NAME']
    resource_group = os.environ['RESOURCE_GROUP']
    subscription_id = os.environ['SUBSCRIPTION_ID']

    spn_credentials = {
        'tenant_id': os.environ['TENANT_ID'],
        'service_principal_id': os.environ['SP_ID'],
        'service_principal_password': os.environ['SP_VALUE'],
    }
    auth = ServicePrincipalAuthentication(
        **spn_credentials)
    workspace = Workspace(
        workspace_name=workspace_name,
        auth=auth,
        subscription_id=subscription_id,
        resource_group=resource_group
    )
    aml_env = create_aml_environment()
    aml_env.register(workspace=workspace)


if __name__ == '__main__':
    main()
