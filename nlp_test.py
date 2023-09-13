# import spacy

# # Load the English language model
# nlp = spacy.load("en_core_web_lg")

# # Define your list of concepts
# concepts = ["headache", "vomiting", "fever"]

# # Create a function to match the input text to concepts
# def match_concept(input_text, concepts):
#     # Process the input text with spaCy
#     doc = nlp(input_text.lower())  # Convert to lowercase for case-insensitive matching

#     # Initialize a dictionary to store concept scores
#     concept_scores = {}

#     # Loop through the concepts and calculate similarity scores
#     for concept in concepts:
#         concept_doc = nlp(concept.lower())  # Convert concept to lowercase
#         similarity_score = doc.similarity(concept_doc)
#         concept_scores[concept] = similarity_score

#     # Find the concept with the highest similarity score
#     best_match = max(concept_scores, key=concept_scores.get)

#     return best_match, concept_scores[best_match]

# # Test the function
# input_text = "i feel nausea, my stomach hurts"

# best_match, similarity_score = match_concept(input_text, concepts)
# print(f"The input text is most likely related to: {best_match} (Similarity Score: {similarity_score})")


import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the English language model with word vectors
nlp = spacy.load("en_core_web_lg")

# The concepts would be the symptoms
concepts = ["headache", "fever"]

# Function to calculate similarity between input text and a list of concepts
def calculate_similarity(input_text, concept_list):
    input_tokens = [token for token in nlp(input_text.lower())]

    concept_similarities = {}

    for concept in concept_list:
        concept_tokens = [token for token in nlp(concept.lower())]

        if not input_tokens or not concept_tokens:
            # Skip empty tokens
            continue

        input_vector = np.mean([token.vector for token in input_tokens], axis=0)
        concept_vector = np.mean([token.vector for token in concept_tokens], axis=0)

        # Calculate cosine similarity
        similarity_score = cosine_similarity([input_vector], [concept_vector])[0][0]
        concept_similarities[concept] = similarity_score

    return concept_similarities

# Test the function with an input text
input_text = "My head hurts"
concept_similarities = calculate_similarity(input_text, concepts)

# Find the concept with the highest similarity score
best_match = max(concept_similarities, key=concept_similarities.get)

print(f"The input text is most likely related to: {best_match}")
print("Concept Similarities:")
for concept, similarity_score in concept_similarities.items():
    print(f"{concept}: {similarity_score:.2f}")
