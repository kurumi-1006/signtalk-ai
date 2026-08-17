import sqlite3
from pathlib import Path
class Outbox:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS outbox (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0)')

    def put(self, event_id: str, payload: str) -> None:
        self.connection.execute('INSERT OR IGNORE INTO outbox(event_id,payload) VALUES (?,?)', (event_id, payload))
        self.connection.commit()

    def pending(self) -> list[tuple[str, str]]: return self.connection.execute('SELECT event_id,payload FROM outbox WHERE sent=0').fetchall()

    def mark_sent(self, event_id: str) -> None:
        self.connection.execute('UPDATE outbox SET sent=1 WHERE event_id=?', (event_id,))
        self.connection.commit()
