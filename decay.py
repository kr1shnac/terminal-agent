import math
from datetime import datetime
from store import get_active, update_confidence, update_last_accessed, archive

ARCHIVE_THRESHOLD = 0.05 #below 5% confidence, a memory gets archived
REINFORCE_BOOST = 0.2 #using a memory adds 0.2 back to its confidence
MIN_DECAY_RATE = 0.02 #decay rate never drops below this, even after lots of reinforcement


def days_since(timestamp_str):
    then = datetime.fromisoformat(timestamp_str)
    now = datetime.utcnow()
    return (now - then).total_seconds() / 86400 #convert secounds to days

def apply_decay():
    for mem in get_active():
        days = days_since(mem["last_accessed_at"])

        if days < 1: #if accessed in hrs then skip
            continue

        new_confidence = mem["confidence_score"] * math.exp(-mem["decay_rate"] * days) #confidence = old_confidence × e^(−decay_rate × days_since_last_used)

        if new_confidence < ARCHIVE_THRESHOLD: #i.e.. less than 5%
            archive(mem["id"])
        else:
            update_confidence(mem["id"], new_confidence)

def reinforce(memory_id, current_confidence, current_decay_rate):
    #Talking about used memory
    #min(1.0) to cap the value within 1 that is like 100%
    #max(MIN_DECAY_RATE) -> 0.02 -> when we use that memory deacy gets boost of 10% so less chance of getting achived
    new_confidence = min(1.0, current_confidence + REINFORCE_BOOST)
    new_decay_rate = max(MIN_DECAY_RATE, current_decay_rate * 0.9)
    update_confidence(memory_id, new_confidence, new_decay_rate)
    update_last_accessed(memory_id)

if __name__ == "__main__":
    apply_decay()
    for mem in get_active():
        print(mem)