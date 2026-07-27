"""
logger.py
---------
Lightweight query logging (Intermediate Enhancement: "Logging user queries").
Writes to a local CSV so it works out of the box on Streamlit Cloud
without needing an external logging service.
"""

import csv
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "query_log.csv")


def log_query(question: str, doc_names: list, num_chunks_found: int, answer: str):
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "documents", "question", "chunks_found", "answer"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            ";".join(doc_names),
            question,
            num_chunks_found,
            answer.replace("\n", " ")[:300],
        ])
