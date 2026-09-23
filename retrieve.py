import re
import math

STOPWORDS = {
    "the", "a", "an", "is", "was", "were",
    "are", "and", "or", "of", "to", "in",
    "on", "for", "with", "it", "this", "that"
}

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) #re.sub(pattern, replacement, text)

    words = text.split()

    result = []

    for word in words:
        if word in STOPWORDS:
            continue

        if len(word) <= 2:
            continue

        result.append(word)

    return result


def term_frequency(words):
    frequencies = {}

    for word in words:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1

    return frequencies


words = tokenize("Python is great. Python is easy to learn.")

print("Tokens:", words)
print("TF:", term_frequency(words))


def document_frequency(memories):
    frequencies = {}

    for memory in memories: #pick one memory in collection of memories
        words = tokenize(memory["text"])
        unique_words = set(words) #remove duplicate words from memory

        for word in unique_words:
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1

    return frequencies


def inverse_document_frequency(memories):
    frequencies = document_frequency(memories)

    total_memories = len(memories)

    idf = {}

    for word in frequencies:
        idf[word] = math.log(total_memories / frequencies[word])

    return idf


def tf_idf(words, idf):
    tf = term_frequency(words)

    vector = {}

    for word in idf:
        vector[word] = tf.get(word, 0) * idf[word]

    return vector


def cosine_similarity(vector_a, vector_b):
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0

    for word in vector_a:
        dot_product += vector_a[word] * vector_b.get(word, 0)
        magnitude_a += vector_a[word] ** 2

    for word in vector_b:
        magnitude_b += vector_b[word] ** 2

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def rank(query, memories):
    idf = inverse_document_frequency(memories)

    query_words = tokenize(query)
    query_vector = tf_idf(query_words, idf)

    results = []

    for memory in memories:
        memory_words = tokenize(memory["text"])
        memory_vector = tf_idf(memory_words, idf)

        score = cosine_similarity(query_vector, memory_vector)

        results.append((memory, score)) #score is stored in index 1

    results.sort(key=lambda item: item[1], reverse=True) #sort using index 1

    return results


memories = [
    {"text": "User is learning Python"},
    {"text": "User likes Python programming"},
    {"text": "User practices Karate"},
    {"text": "User is learning machine learning"}
]

print("DF:")
print(document_frequency(memories))

print("\nIDF:")
print(inverse_document_frequency(memories))