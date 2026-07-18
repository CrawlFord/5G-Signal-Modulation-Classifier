import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import PyTorchModelHubMixin


MODULATION_TYPES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

class ModulationClassifier(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="modulation-classifier",
    pipeline_tag="tabular-classification",
    license="cc-by-nc-sa-4.0",
):
    def __init__(self, num_classes: int = 11, input_channels: int = 2, input_length: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=8, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_length // 4), 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x
model = ModulationClassifier.from_pretrained("tandrewu1/5G-Signal-Modulation-Classifier")
model.eval()


def predict(iq_text):
    try:
        lines = iq_text.strip().split("\n")
        i_samples = [float(x) for x in lines[0].split(",")]
        q_samples = [float(x) for x in lines[1].split(",")]

        if len(i_samples) != 128 or len(q_samples) != 128:
            return {"Error": 1.0}

        iq_data = np.array([[i_samples, q_samples]], dtype=np.float32)
        tensor_input = torch.tensor(iq_data)

        with torch.no_grad():
            output = model(tensor_input)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        result = {MODULATION_TYPES[i]: prob.item() for i, prob in enumerate(probabilities)}
        return result

    except Exception as e:
        return {"Error": 1.0}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="IQ Samples (line 1: 128 I values comma-separated, line 2: 128 Q values comma-separated)",
        lines=4,
        placeholder="0.01, -0.03, 0.05, ...\n0.02, 0.01, -0.04, ...",
    ),
    outputs=gr.Label(label="Predicted Modulation", num_top_classes=5),
    title="5G Signal Modulation Classifier",
    description="Classify radio signal modulation type from raw IQ samples. Paste 128 I-channel values on line 1 and 128 Q-channel values on line 2.",
)

if __name__ == "__main__":
    demo.launch()