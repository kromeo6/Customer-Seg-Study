import os
from azureml.core import ScriptRunConfig, Experiment, Environment
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException


__here__ = os.path.dirname(__file__)

COMPUTE = 'cpu-compute'
ENVIRONMENT = 'env'
EXPERIMENT = 'Customer-segm-exp'


class ConnectAml:
    def __init__(self, spn_credentials, subscription_id,
                 workspace_name, resource_group):
        auth = ServicePrincipalAuthentication(
            **spn_credentials
        )
        self.workspace = Workspace(
            workspace_name=workspace_name,
            auth=auth,
            subscription_id=subscription_id,
            resource_group=resource_group
        )


    def get_compute(self, compute_name, vm_size=None):
        try:
            compute_target = ComputeTarget(
                workspace=self.workspace,
                name=compute_name
            )
        except ComputeTargetException:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                min_nodes=1,
                max_nodes=2
            )
            compute_target = ComputeTarget.create(
                self.workspace,
                compute_name,
                compute_config
            )
            compute_target.wait_for_completion(
                show_output=True,
                timeout_in_minutes=20
            )
        return compute_target


def run(aml_interface, arg_val):
    experiment = Experiment(aml_interface.workspace, EXPERIMENT)
    src_dir = __here__
    run_config = ScriptRunConfig(
        source_directory=src_dir,
        script='train.py',
        arguments=['--arg1', arg_val]
    )
    run_config.run_config.target = aml_interface.get_compute(
        COMPUTE,
        'STANDARD_D2_V2'
    )
    aml_run_env = Environment.get(
        aml_interface.workspace,
        ENVIRONMENT
    )
    run_config.run_config.environment = aml_run_env
    run = experiment.submit(config=run_config)
    run.wait_for_completion(show_output=True)


def main():
    # Retrieve vars from env
    workspace_name = os.environ['AML_WORKSPACE_NAME']
    resource_group = os.environ['RESOURCE_GROUP']
    subscription_id = os.environ['SUBSCRIPTION_ID']

    spn_credentials = {
        'tenant_id': os.environ['TENANT_ID'],
        'service_principal_id': os.environ['SP_ID'],
        'service_principal_password': os.environ['SP_VALUE'],
    }

    aml_interface = ConnectAml(
        spn_credentials, subscription_id, workspace_name, resource_group
    )
    arg_val = os.environ['TENANT_ID']
    run(aml_interface, arg_val=arg_val)


if __name__ == '__main__':
    main()
    