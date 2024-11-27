import gradio as gr
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoModelForMaskedLM
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import torch

yolo_weights_path = "final_wts.pt"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
trocr_model.config.num_beams = 2

yolo_model = YOLO(yolo_weights_path).to(device)
roberta_model = AutoModelForMaskedLM.from_pretrained("roberta-large").to(device)


print(f'TrOCR, YOLO and Roberta Models loaded on {device}')

CONFIDENCE_THRESHOLD = 0.72
BLEU_THRESHOLD = 0.6


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
        output = trocr_model.generate(pixel_values, return_dict_in_generate=True, output_logits=True, max_new_tokens=30)
        generated_texts = processor.batch_decode(output.sequences, skip_special_tokens=True)
        generated_tokens = [processor.tokenizer.convert_ids_to_tokens(seq) for seq in output.sequences]
        stacked_logits = torch.stack(output.logits, dim=1)
        return generated_texts, stacked_logits, generated_tokens

    def get_scores(logits):
        scores = logits.softmax(-1).max(-1).values.mean(-1)
        return scores

    def post_process_texts(generated_texts):
        for i in range(len(generated_texts)):
            if len(generated_texts[i]) > 2 and generated_texts[i][:2] == '# ':
                generated_texts[i] = generated_texts[i][2:]

            if len(generated_texts[i]) > 2 and generated_texts[i][-2:] == ' #':
                generated_texts[i] = generated_texts[i][:-2]
        return generated_texts

    def get_qualified_texts(generated_texts, scores, y, logits, tokens):
        qualified_texts = []
        for text, score, y_i, logits_i, tokens_i in zip(generated_texts, scores, y, logits, tokens):
            if score > CONFIDENCE_THRESHOLD:
                qualified_texts.append({
                    'text': text,
                    'score': score,
                    'y': y_i,
                    'logits': logits_i,
                    'tokens': tokens_i
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

    def get_lm_logits(ocr_tokens, confidence):
        tokens = ocr_tokens.clone()
        indices = torch.where(confidence < 0.5)
        for i, j in zip(indices[0], indices[1]):
            if i != 6:
                continue
            tokens[i, j] = torch.tensor(50264)
        inputs = tokens.reshape(1, -1)
        with torch.no_grad():
            outputs = roberta_model(input_ids=inputs, attention_mask=torch.ones(inputs.shape).to(device))
            lm_logits = outputs.logits
        return lm_logits.reshape(ocr_tokens.shape[0], ocr_tokens.shape[1], -1), indices



    cropped_images, y, bounding_box_path = get_cropped_images(image_path)
    if debug:
        print('Number of cropped images:', len(cropped_images))
    generated_texts, logits, gen_tokens = get_model_output(cropped_images)
    normalised_scores = get_scores(logits)
    generated_df = pd.DataFrame({
        'text': generated_texts,
    })
    if return_texts == 'generated':
        return pd.DataFrame({
            'text': generated_texts,
            'score': normalised_scores,
            'y': y,
        })
    generated_texts = post_process_texts(generated_texts)
    if return_texts == 'post_processed':
        return pd.DataFrame({
            'text': generated_texts,
            'score': normalised_scores,
            'y': y
        })
    qualified_texts = get_qualified_texts(generated_texts, normalised_scores, y, logits, gen_tokens)
    if return_texts == 'qualified':
        return pd.DataFrame(qualified_texts)
    qualified_texts = get_adjacent_bleu_scores(qualified_texts)
    if return_texts == 'qualified_with_bleu':
        return pd.DataFrame(qualified_texts)
    final_texts = remove_overlapping_texts(qualified_texts)
    final_texts_df = pd.DataFrame(final_texts, columns=['text', 'score', 'y'])
    final_logits = [text['logits'] for text in final_texts]
    logits = torch.stack([logit for logit in final_logits], dim=0)
    tokens = logits.argmax(-1)
    confidence = logits.softmax(-1).max(-1).values
    if return_texts == 'final':
        return final_texts_df

    lm_logits, indices = get_lm_logits(tokens, confidence)
    combined_logits = logits.clone()
    for i, j in zip(indices[0], indices[1]):
        combined_logits[i, j] = logits[i, j] * 0.9 + lm_logits[i, j] * 0.1

    return final_texts_df, bounding_box_path, tokens, combined_logits, confidence, generated_df




def process_image(image):
    text, bounding_path = "", ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        image.save(temp_image.name)
        image_path = temp_image.name
        df, bounding_path, tokens, logits, confidence, generated_df = inference(image_path, debug=False, return_texts='final_v2')
        text = df['text'].str.cat(sep='\n')
        before_text = generated_df['text'].str.cat(sep='\n')
    bounding_img = Image.open(bounding_path)
    return bounding_img, before_text, text

# Define Gradio Interface
interface = gr.Interface(
    fn=process_image,  # Call the process_image function
    inputs=gr.Image(type="pil"),  # Expect an image input
    outputs=[
        gr.Image(type="pil", label="Bounding Box Image"),
        gr.Textbox(label="Extracted Text (Custom trained YOLO Object Detection + TrOCR Vision Transformer)"),
        gr.Textbox(label="Post Processed Text (BLEU score based filtering + Roberta contextual understanding)"),
    ],
    title="OCR Pipeline with YOLO, TrOCR and Roberta",
    description="Upload an image to detect text regions with YOLO, merge bounding boxes, and extract text using TrOCR which is then preprocessed with Roberta for contextual understanding.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)