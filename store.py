import sqlite3
from datetime import datetime

con = sqlite3.connect("app.db")
con.row_factory = sqlite3.Row #let us read cloumn by names, like json or dict

con.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY,
        text TEXT,
        memory_type TEXT,
        confidence_score REAL,
        decay_rate REAL,
        created_at TEXT,
        last_accessed_at TEXT,
        is_archived INTEGER
    )
""")


def insert_memory(text, memory_type):
    now = datetime.utcnow().isoformat()

    if memory_type == "fact":
        decay_rate = 0.2
    else:
        decay_rate = 0.1

    con.execute("""
        INSERT INTO memory (text, memory_type, confidence_score, decay_rate, created_at, last_accessed_at, is_archived)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (text, memory_type, 1.0, decay_rate, now, now, 0))

    con.commit()


def get_active():
    rows = con.execute("SELECT * FROM memory WHERE is_archived = 0").fetchall() #insted of using for loop iteration, fetchatt() will help you return all rows at once
    
    result = []

    for row in rows:
        result.append(dict(row))

    return result
    #return [dict(row) for row in rows]


def update_confidence(memory_id, new_confidence, new_decay_rate=None):
    if new_decay_rate is not None:
        con.execute("UPDATE memory SET confidence_score = ?, decay_rate = ? WHERE id = ?",
        (new_confidence, new_decay_rate, memory_id)
        )
    else:
        con.execute(
            "UPDATE memory SET confidence_score = ? WHERE id = ?",
            (new_confidence, memory_id)
        )
    con.commit()


def update_last_accessed(memory_id):
    now = datetime.utcnow().isoformat()
    con.execute(
        "UPDATE memory SET last_accessed_at = ? WHERE id = ?", (now, memory_id)
    )
    con.commit()


def archive(memory_id):
    con.execute(
        "UPDATE memory SET is_archived = 1 WHERE id = ?", (memory_id)
    )
    con.commit()


if __name__ == "__main__": #this block only runs when you execute this file directly (python store.py)
    insert_memory("User's name is Krishna C", "fact")
    insert_memory("Debugging TF-IDF bug today", "event")

    for memory in get_active():
        print(memory)

    con.close()

