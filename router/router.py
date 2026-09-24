from router.rules import rule_based_tier


MODEL_TIERS = {
    "cheap": "poolside/laguna-s-2.1:free",

    # Same model for v1.
    # The important thing right now is proving the routing mechanism.
    "powerful": "poolside/laguna-s-2.1:free",
}


def route(query):
    tier, score = rule_based_tier(query)

    model = MODEL_TIERS[tier]

    reason = f"rule-based: score {score} -> {tier}"

    return model, tier, reason