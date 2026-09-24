TECH_KEYWORDS = {
    "implement", "debug", "architect", "analyze", "optimize",
    "design", "compare", "explain", "refactor", "algorithm",
    "function", "class", "error", "build", "write", "fix"
}

def score_complexity(query):
    score = 0
    words = query.split()

    #signal 1 -> query length
    word_count = len(words)

    if word_count > 50:
        score += 3
    elif word_count > 20:
        score += 2
    elif word_count > 10:
        score += 1

    #signal 2 -> technical keywords
    matches = sum(
        1 for w in words
        if w.lower() in TECH_KEYWORDS
    )

    score += min(matches, 3)

    #signal 3 -> code / multiline
    if "```" in query or query.count("\n") > 2:
        score += 1

    return score

def rule_based_tier(query, threshold=5):
    score = score_complexity(query)

    tier = "powerful" if score >= threshold else "cheap"

    return tier, score