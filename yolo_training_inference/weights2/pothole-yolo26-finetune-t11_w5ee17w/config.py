import os
from truss_train import definitions
from truss.base.truss_config import Accelerator, AcceleratorSpec

training_job = definitions.TrainingJob(
    name="pothole-yolo26-finetune-t11",
    image=definitions.Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=definitions.Compute(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=1)
    ),
    runtime=definitions.Runtime(
        start_commands=[
            "pip install -r requirements.txt",
            "pip install --force-reinstall --no-deps opencv-python-headless",
            "python train.py",
        ],
        environment_variables={
            "ROBOFLOW_API_KEY": os.environ["ROBOFLOW_API_KEY"],
            "CHECKPOINT_PATH": "/b10/workspace/checkpoints",
        },
        checkpointing_config=definitions.CheckpointingConfig(
            enabled=True,
            checkpoint_path="/b10/workspace/checkpoints",
        ),
    ),
)

training_project = definitions.TrainingProject(
    name="pothole-yolo26-finetune-t11",
    job=training_job,
    team_name="11",
)