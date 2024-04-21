import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
import Levenshtein
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
# Function to load GloVe vectors
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_glove_vectors(glove_file_path):
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        glove_model = {}
        for line in file:
            parts = line.split()
            word = parts[0]
            try:
                # Attempt to convert the rest of the line to floats, and skip the line if it fails
                embedding = np.array([float(val) for val in parts[1:]])
            except ValueError:
                continue  # Skip lines that don't have the correct format
            glove_model[word] = embedding
        return glove_model
def vectorize_document(doc, glove_model):
    vectors = [glove_model[word] for word in doc.lower().split() if word in glove_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(len(next(iter(glove_model.values()))))  # Return a zero vector if no words matched

# Continue with the rest of yo


def calculate_similarity(method, sentence1, sentence2):
    start_time = time.time()
    similarity = None

    # Preprocess the sentences to lowercase
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    if method == 'cosine':
        vectorizer = TfidfVectorizer().fit([sentence1, sentence2])
        vector1, vector2 = vectorizer.transform([sentence1, sentence2])
        similarity = cosine_similarity(vector1, vector2)[0, 0]

    elif method == 'jaccard':
        set1 = set(ngrams(sentence1, n=2))
        set2 = set(ngrams(sentence2, n=2))
        similarity = 1 - jaccard_distance(set1, set2)

    elif method == 'euclidean':
        vectorizer = TfidfVectorizer().fit([sentence1, sentence2])
        vector1, vector2 = vectorizer.transform([sentence1, sentence2])
        similarity = 1 / (1 + euclidean_distances(vector1, vector2)[0, 0])

    elif method == 'levenshtein':
        similarity = 1 - (Levenshtein.distance(sentence1, sentence2) / max(len(sentence1), len(sentence2)))

    elif method == 'sentence_transformer':
        sentences = [sentence1, sentence2]
        # Tokenize sentences
        encoded_input = sentence_transformer_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = sentence_transformer_model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        similarity = cosine_similarity(sentence_embeddings)[0][1]

    elif method == 'glove':
        sentences = [sentence1, sentence2]
        vectors = [vectorize_document(doc, glove_model) for doc in sentences]
        similarity = cosine_similarity(vectors)[0][1]

    elapsed_time = time.time() - start_time
    return similarity, elapsed_time



def aggregate_sentences(sentences):
    return ' '.join(sentences)


def calculate_similarity_with_aggregation(df, method='levenshtein'):
    num_rows = df.shape[0]
    num_columns = 40  # Assuming you have 40 columns named 'CoT_0' to 'CoT_39'
    similarity_matrices = np.zeros((num_rows, num_columns))
    start_time = time.time()

    for index, row in df.iterrows():
        aggregated_sentence = ''
        for i in range(1, 39):
            current_sentence = row[f'CoT_{i}']
            previous_sentence = row[f'CoT_{i - 1}']

            aggregated_sentence += previous_sentence + ' '
            similarity = calculate_similarity(method, aggregated_sentence, current_sentence)
            similarity_matrices[index, i] = similarity[0]
    elapsed_time = time.time() - start_time
    print(f'{method} with aggregation time cost: {elapsed_time}s')
    return similarity_matrices


def calculate_similarity_pairwise(df, method='levenshtein'):
    # Assuming 'calculate_similarity' is a function you've defined that
    # calculates the similarity between two sentences.
    start_time = time.time()
    # Number of rows and columns
    num_rows = df.shape[0]
    num_columns = 40  # Assuming you have 40 columns named 'CoT_0' to 'CoT_39'

    # Initialize a 3D numpy array to hold the similarity matrices for each row
    # Shape will be (number of rows) x (40 columns) x (40 columns)
    similarity_matrices = np.zeros((num_rows, num_columns))

    # Iterate over each row and column pair to calculate pairwise similarity
    for index, row in df.iterrows():  # Loop through rows
        skip_doc_idx = []
        for i in range(1, num_columns):  # Loop through columns for the current row
            sim_buffer = []
            for j in range(0, i):  # Loop to fill the upper triangle
                # if (i in skip_doc_idx) or (j in skip_doc_idx):
                #     similarity_matrices[index, i, j] = 1
                #     similarity_matrices[index, j, i] = 1  # Fill the symmetric value
                #     continue
                current_sentence = row[f'CoT_{i}']
                other_sentence = row[f'CoT_{j}']

                # Calculate similarity and fill both [i, j] and [j, i] to maintain symmetry
                similarity = calculate_similarity(method, current_sentence, other_sentence)

                sim_buffer.append(similarity[0])
            similarity_matrices[index, i] = np.mean(sim_buffer)

    elapsed_time = time.time() - start_time
    print(f'{method} with pairwise time cost: {elapsed_time}')
    return similarity_matrices

def calculate_similarity_with_bigram(df, method='levenshtein'):
    num_rows = df.shape[0]
    num_columns = 40  # Assuming you have 40 columns named 'CoT_0' to 'CoT_39'
    similarity_matrices = np.zeros((num_rows, num_columns))
    start_time = time.time()

    for index, row in df.iterrows():
        for i in range(1, 39):
            current_sentence = row[f'CoT_{i}']
            previous_sentence = row[f'CoT_{i - 1}']
            similarity = calculate_similarity(method, previous_sentence, current_sentence)
            similarity_matrices[index, i] = similarity[0]
    elapsed_time = time.time() - start_time
    print(f'{method} with bigram time cost: {elapsed_time}s')
    return similarity_matrices


def check_correctness_MC(correct_answer,cot_answers,question):
    correctness_list = []
    for i in range(len(cot_answers)):
        correct = correct_answer
        cot = cot_answers[i]
        q = question
        cot = str(cot)
        if (cot == correct):
            correctness_list.append(1)
            continue
        # Extract the options and their corresponding values from the question
        options = {}
        for option in ['a', 'b', 'c', 'd', 'e']:
            if option + ' )' in q:
                value = q.split(option + ' )')[-1].strip().split(',')[0].strip()
                options[option] = value
        if options[correct] in cot:
            correctness_list.append(1)
        else:
            correctness_list.append(0)

    return correctness_list

if __name__ == '__main__':
    # sentence_transformer_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # sentence_transformer_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # Load GloVe vectors (adjust the path to where you've stored your GloVe file)
    # glove_model = load_glove_vectors(
    #     '../../src/glove.840B.300d/glove.840B.300d.txt')  # Use the correct path to your GloVe file
    methods_list = ['cosine', 'jaccard', 'euclidean', 'levenshtein', 'glove', 'sentence_transformer']
    hard = pd.read_csv('../data/Evaluation_CoTs/gpt-3.5-turbo-0125/BigBench_hard.csv')
    # Example usage:
    # Assuming your DataFrame is named 'df' and has columns CoT_0 to CoT_39
    similarities = calculate_similarity_with_bigram(hard,method='jaccard')
    # print("Similarities:", similarities[0])
    print(similarities.shape)