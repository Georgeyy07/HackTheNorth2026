import os
from truss_train import definitions
from truss.base.truss_config import Accelerator, AcceleratorSpec

training_job = definitions.TrainingJob(
    name="pothole-yolo26-finetune",
    image=definitions.Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=definitions.Compute(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=1)
    ),
    runtime=definitions.Runtime(
        start_commands=["python train.py"],
        environment_variables={
            "ROBOFLOW_API_KEY": os.environ["ROBOFLOW_API_KEY"],
        },
    ),
)