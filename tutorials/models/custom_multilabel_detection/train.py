from pyannote.audio import Trainer
from pyannote.database import get_protocol

protocol = get_protocol("ChildLens.SpeakerDiarization.audio")
trainer = Trainer(
    config_file="config.yaml",
    protocol=protocol,
    model_dir="models/",
    devices=1,  # Use GPU if available
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=200,
    num_workers=4
)
trainer.fit()