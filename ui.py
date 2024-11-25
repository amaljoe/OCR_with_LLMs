import gradio as gr
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Initialize TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate intersection
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def group_boxes(boxes, iou_threshold=0.5):
    """Group bounding boxes that overlap into a single bounding box."""
    merged_boxes = []
    while boxes:
        # Take the first box
        main_box = boxes.pop(0)
        to_merge = [main_box]

        # Check for overlap with other boxes
        for box in boxes[:]:
            if iou(main_box, box) > iou_threshold:
                to_merge.append(box)
                boxes.remove(box)

        # Merge all overlapping boxes
        x1 = min(box[0] for box in to_merge)
        y1 = min(box[1] for box in to_merge)
        x2 = max(box[2] for box in to_merge)
        y2 = max(box[3] for box in to_merge)
        merged_boxes.append([x1, y1, x2, y2])

    return merged_boxes

def crop_and_recognize(image, bounding_boxes):
    """Crop image regions based on YOLO boxes, merge boxes, and perform OCR."""
    image_np = np.array(image)

    # Merge overlapping boxes
    merged_boxes = group_boxes(bounding_boxes)

    results = []
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        # Crop the region
        cropped_region = image_np[int(y1):int(y2), int(x1):int(x2)]
        cropped_image = Image.fromarray(cropped_region)

        # Perform OCR using TrOCR
        pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        text = processor.decode(generated_ids[0], skip_special_tokens=True)
        results.append({"text": text, "box": box})

    return results

def process_image(image):
    """
    Process the image, run YOLO for bounding box detection (mock example),
    and run OCR on detected text regions.
    """
    # Example bounding boxes from YOLO (x_min, y_min, x_max, y_max)
    bounding_boxes = [
        [5, 50, 20, 15],  # Box 1
        [21, 10, 35, 15], # Box 2 (same line as Box 1)
        [50, 20, 20, 25],  # Box 3 (new line)
    ]

    # Perform OCR on grouped boxes
    ocr_results = crop_and_recognize(image, bounding_boxes)

    # Return text results for UI display
    return "\n".join([f"Text: '{result['text']}', Box: {result['box']}" for result in ocr_results])

# Define Gradio Interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="OCR Pipeline with YOLO and TrOCR",
    description="Upload an image to detect text regions with YOLO, merge bounding boxes, and extract text using TrOCR.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)