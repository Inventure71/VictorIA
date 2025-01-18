from segment_anything import sam_model_registry, SamPredictor

class SamModelHandler:
    def __init__(self, checkpoint_path, model_type="vit_h", device="mps"):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict(self, points, labels, multimask_output=True):
        return self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output
        )
