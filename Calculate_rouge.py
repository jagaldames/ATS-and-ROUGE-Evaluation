from nltk.util import ngrams
from nltk import word_tokenize


# Remove special characters, punctuation, and extra white spaces
def preprocess_text(text):
    text = text.lower()
    text = "".join(c for c in text if c.isalnum() or c.isspace())
    text = " ".join(text.split())
    return text

# Read reference and summary texts from files
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def calculate_rouge_c(reference, summary, n):
    reference = preprocess_text(reference)
    summary = preprocess_text(summary)
    
    # Generate n-grams for reference and summary
    reference_ngrams = set(ngrams(word_tokenize(reference, language='spanish'), n))
    summary_ngrams = set(ngrams(word_tokenize(summary, language='spanish'), n))
    
    # Calculate the count of matching n-grams
    matching_ngrams = len(reference_ngrams.intersection(summary_ngrams))
    
    # Calculate recall, precision, and F1 scores
    recall = matching_ngrams / len(reference_ngrams)
    precision = matching_ngrams / len(summary_ngrams)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return recall, precision, f1_score

# Create matrix to search for the longest common sequence
def lcs_length(X, Y):
    m = len(X)
    n = len(Y)

    c = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i - 1][j], c[i][j - 1])

    return c[m][n]

def rouge_l(reference, candidate):
    reference_tokens = word_tokenize(preprocess_text(reference), language='spanish')
    candidate_tokens = word_tokenize(preprocess_text(candidate), language='spanish')

    lcs = lcs_length(reference_tokens, candidate_tokens)

    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)

    rouge_l_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return rouge_l_score, precision, recall, lcs


hyp_path = './Texts/00 - AC01.txt'

ref_paths = [
    './Texts/01 - Lex Rank.txt',
    './Texts/02 - Text Rank.txt',
    './Texts/03 - LSA.txt',
    './Texts/04 - Luhn.txt',
    './Texts/05 - GPT.txt' #,    './Texts/06 - GPT Ex.txt'
]

reference_text = read_text_from_file(hyp_path)

n_gram = 1
for ref_path in ref_paths:
    print(ref_path)
    for n_gram in range(1,3):
        summary_text = read_text_from_file(ref_path)
        recall, precision, f1_score = calculate_rouge_c(reference_text, summary_text, n_gram)
        print(f"\tROUGE-C-{n_gram} Score: Precision: {precision: .5f}, Recall: {recall: .5f}, F1 Score: {f1_score: .5f}" )
    f1_score, precision, recall, lcs = rouge_l(reference_text, summary_text)
    print(f"\tROUGE-C-L Score: Precision: {precision: .5f}, Recall: {recall: .5f}, F1 Score: {f1_score: .5f}, LCS: {lcs}")
    