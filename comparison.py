from docx import Document
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import editdistance

# Tokenize text
def tokenize_text(text):
    return nltk.word_tokenize(text)

# Read Word files
def read_word_file(file_path):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip() != ""])
    return text

# Calculate Character Error Rate (CER)
def calculate_cer(reference, hypothesis):
    reference = reference.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")
    edit_distance = editdistance.eval(reference, hypothesis)
    cer = edit_distance / len(reference) if len(reference) > 0 else 0
    return cer

# Create binary labels for evaluation
def create_binary_labels(actual_tokens, predicted_tokens):
    # Combine actual and predicted tokens to ensure both have the same reference
    all_tokens = sorted(set(actual_tokens + predicted_tokens))

    # Create binary labels for both actual and predicted tokens
    y_true = [1 if token in actual_tokens else 0 for token in all_tokens]
    y_pred = [1 if token in predicted_tokens else 0 for token in all_tokens]

     # Debugging
    print(f"All Tokens: {all_tokens}")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    return y_true, y_pred

# Calculate BLEU score
def calculate_bleu(actual_tokens, predicted_tokens):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([actual_tokens], predicted_tokens, smoothing_function=smoothie)

def calculate_jaccard(actual_tokens, predicted_tokens):
    actual_set = set(actual_tokens)
    predicted_set = set(predicted_tokens)
    intersection = len(actual_set & predicted_set)
    union = len(actual_set | predicted_set)
    jaccard = intersection / union if union > 0 else 0
    return jaccard


# File paths
actual_file = "C:\\Users\\Akansha\\OneDrive\\Desktop\\IITB\\SEM 1\\fml\\project\\actual.docx"
predicted_file = "C:\\Users\\Akansha\\OneDrive\\Desktop\\IITB\\SEM 1\\fml\\project\\prediction.docx"

# Read and tokenize data
actual_text = read_word_file(actual_file)
predicted_text = read_word_file(predicted_file)

# Split text into paragraphs
actual_paragraphs = actual_text.split(".\n")
predicted_paragraphs = predicted_text.split(".\n")

# Ensure the same number of paragraphs
num_paragraphs = max(len(actual_paragraphs), len(predicted_paragraphs))
actual_paragraphs.extend([""] * (num_paragraphs - len(actual_paragraphs)))
predicted_paragraphs.extend([""] * (num_paragraphs - len(predicted_paragraphs)))

# Initialize accumulators for scores
total_precision, total_recall, total_f1, total_cer, total_bleu = 0, 0, 0, 0, 0

# Process each paragraph
for i, (actual_paragraph, predicted_paragraph) in enumerate(zip(actual_paragraphs, predicted_paragraphs)):
    print(f"\n=== Paragraph {i + 1} ===")
    print(f"Actual: {actual_paragraph}")
    print(f"Predicted: {predicted_paragraph}")

    actual_tokens = tokenize_text(actual_paragraph)
    predicted_tokens = tokenize_text(predicted_paragraph)

    # Calculate CER
    cer = calculate_cer(actual_paragraph, predicted_paragraph)
    total_cer += cer

    # Create labels for precision, recall, and F1 score
    y_true, y_pred = create_binary_labels(actual_tokens, predicted_tokens)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    total_precision += precision
    total_recall += recall
    total_f1 += f1

    # Calculate BLEU
    bleu = calculate_bleu(actual_tokens, predicted_tokens)
    total_bleu += bleu

    print(f"\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, CER: {cer:.4f}, BLEU: {bleu:.4f}")

# Calculate averages
num_valid_paragraphs = len(actual_paragraphs)
avg_precision = total_precision / num_valid_paragraphs
avg_recall = total_recall / num_valid_paragraphs
avg_f1 = total_f1 / num_valid_paragraphs
avg_cer = total_cer / num_valid_paragraphs
avg_bleu = total_bleu / num_valid_paragraphs

print("\n=== Final Averages ===")
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
print(f"Average F1 Score: {avg_f1:.2f}")
print(f"Average CER: {avg_cer:.4f}")
print(f"Average BLEU: {avg_bleu:.4f}")

# Calculate Jaccard similarity
jaccard = calculate_jaccard(actual_tokens, predicted_tokens)

# Print additional metrics
print("\nadditional\n")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, Jaccard: {jaccard:.2f}")

