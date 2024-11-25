import gradio as gr
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import torch

yolo_weights_path = "runs/detect/train103/weights/last.pt"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
trocr_model.config.num_beams = 2

yolo_model = YOLO(yolo_weights_path).to(device)

print(f'TrOCR and YOLO Models loaded on {device}')

CONFIDENCE_THRESHOLD = 0.72
BLEU_THRESHOLD = 0.6


def inference(image_path, debug=False, return_texts='final'):
    def get_cropped_images(image_path):
        results = yolo_model(image_path, save=True)
        patches = []
        ys = []
        for box in sorted(results[0].boxes, key=lambda x: x.xywh[0][1]):
            image = Image.open(image_path).convert("RGB")
            x_center, y_center, w, h  = box.xywh[0].cpu().numpy()
            x, y = x_center - w / 2, y_center - h / 2
            cropped_image = image.crop((x, y, x + w, y + h))
            patches.append(cropped_image)
            ys.append(y)
        bounding_box_path = results[0].save_dir + results[0].path[results[0].path.rindex('/'):-4] + '.jpg'
        return patches, ys, bounding_box_path

    def get_model_output(images):
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        output = trocr_model.generate(pixel_values, return_dict_in_generate=True, output_scores=True, max_new_tokens=30)
        generated_texts = processor.batch_decode(output.sequences, skip_special_tokens=True)
        return generated_texts, output.sequences_scores

    def post_process_texts(generated_texts):
        for i in range(len(generated_texts)):
            if len(generated_texts[i]) > 2 and generated_texts[i][:2] == '# ':
                generated_texts[i] = generated_texts[i][2:]

            if len(generated_texts[i]) > 2 and generated_texts[i][-2:] == ' #':
                generated_texts[i] = generated_texts[i][:-2]
        return generated_texts

    def get_qualified_texts(generated_texts, scores, y):
        qualified_texts = []
        for text, score, y_i in zip(generated_texts, scores, y):
            if score > CONFIDENCE_THRESHOLD:
                qualified_texts.append({
                    'text': text,
                    'score': score,
                    'y': y_i
                })
        return qualified_texts

    def get_adjacent_bleu_scores(qualified_texts):
        def get_bleu_score(hypothesis, references):
            weights = [0.5, 0.5]
            smoothing = SmoothingFunction()
            return bleu_score.sentence_bleu(references, hypothesis, weights=weights,
                                            smoothing_function=smoothing.method1)

        for i in range(len(qualified_texts)):
            hyp = qualified_texts[i]['text'].split()
            bleu = 0
            if i < len(qualified_texts) - 1:
                ref = qualified_texts[i + 1]['text'].split()
                bleu = get_bleu_score(hyp, [ref])
            qualified_texts[i]['bleu'] = bleu
        return qualified_texts

    def remove_overlapping_texts(qualified_texts):
        final_texts = []
        new = True
        for i in range(len(qualified_texts)):
            if new:
                final_texts.append(qualified_texts[i])
            else:
                if final_texts[-1]['score'] < qualified_texts[i]['score']:
                    final_texts[-1] = qualified_texts[i]
            new = qualified_texts[i]['bleu'] < BLEU_THRESHOLD
        return final_texts

    cropped_images, y, bounding_box_path = get_cropped_images(image_path)
    if debug:
        print('Number of cropped images:', len(cropped_images))
    generated_texts, scores = get_model_output(cropped_images)
    normalised_scores = np.exp(scores.to('cpu').numpy())
    if return_texts == 'generated':
        return pd.DataFrame({
            'text': generated_texts,
            'score': normalised_scores,
            'y': y
        })
    generated_texts = post_process_texts(generated_texts)
    if return_texts == 'post_processed':
        return pd.DataFrame({
            'text': generated_texts,
            'score': normalised_scores,
            'y': y
        })
    qualified_texts = get_qualified_texts(generated_texts, normalised_scores, y)
    if return_texts == 'qualified':
        return pd.DataFrame(qualified_texts)
    qualified_texts = get_adjacent_bleu_scores(qualified_texts)
    if return_texts == 'qualified_with_bleu':
        return pd.DataFrame(qualified_texts)
    final_texts = remove_overlapping_texts(qualified_texts)
    final_texts_df = pd.DataFrame(final_texts, columns=['text', 'score', 'y'])
    return final_texts_df, bounding_box_path


# image_path = "yolo_dataset/val/n06-175.png"
# df, bounding_path = inference(image_path, debug=False, return_texts='final')
# bounding_paths

def process_image(image):
    text, bounding_path = "", ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        image.save(temp_image.name)
        image_path = temp_image.name
        df, bounding_path = inference(image_path, debug=False, return_texts='final')
        text = df['text'].str.cat(sep='\n')
    return text

# Define Gradio Interface
interface = gr.Interface(
    fn=process_image,  # Call the process_image function
    inputs=gr.Image(type="pil"),  # Expect an image input
    outputs="text",
    title="OCR Pipeline with YOLO and TrOCR",
    description="Upload an image to detect text regions with YOLO, merge bounding boxes, and extract text using TrOCR.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)