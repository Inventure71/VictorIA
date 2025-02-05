import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Sam2ModelHandler:
    """
    Simple wrapper class around the SAM 2 image predictor.
    """

    def __init__(self, config_path, checkpoint_path):
        if torch.mps.is_available():
            device = "mps"
        # Optionally check for CUDA here.
        else:
            device = "cpu"

        print(f"Using device: {device}")
        self.device = device

        # Build the SAM 2 model
        self.model = build_sam2(config_path, checkpoint_path, device=device)
        self.model.to(device=self.device)

        # Create an image predictor
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image):
        """
        Sets an image for SAM 2 to perform segmentation on.
        """
        self.predictor.set_image(image)

    def predict(self, points, labels, multimask_output=True):
        # Convert the inputs to NumPy arrays with explicit types.
        points = np.array(points, dtype=np.float32)  # Expect shape (N, 2)
        labels = np.array(labels, dtype=np.int32)  # Expect shape (N,)

        # Option 1: Unpack the dictionary
        input_prompts = {
            "point_coords": points,
            "point_labels": labels
        }

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                **input_prompts,  # Unpacking the dict here
                multimask_output=multimask_output
            )
        return masks, scores, logits

