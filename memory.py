from store import (
    insert_memory,
    get_active,
    update_confidence,
    update_last_accessed,
    archive
)

from retrieve import rank

from decay import apply_decay, reinforce

class Memory:

    def __init__(self):
        apply_decay()

    def add(self, text, memory_type="fact"):
        return insert_memory(text, memory_type)

    def search(self, query, top_k=5):
        ranked = rank(query, get_active())

        results = []

        for mem, score in ranked:
            if score > 0:
                results.append(mem)

        results = results[:top_k]

        for mem in results:
            reinforce(
                mem["id"],
                mem["confidence_score"],
                mem["decay_rate"]
            )

        return results

    def get_all(self):
        return get_active()
