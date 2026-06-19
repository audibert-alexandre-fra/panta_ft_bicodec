""" Simple Data Base """
import json
from pathlib import Path
import sqlite3
from typing import Any
from panta_audio_preprocessing.panta_audio.panta_audio import SegmentRecord
import time


def save_json(path: str, data: dict[str, Any]):
    if not(path.endswith(".json")):
        path += ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

class PantaDB:
    def __init__(self, db_path: str | Path) -> None:
        """ Constructor of our database0"""
        self.db_path: Path = Path(db_path)
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path, timeout=30.0)
        self._create_tables()
    
    def _create_tables(self) -> None:
        """ Construction of our table called segments """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id                  TEXT PRIMARY KEY,
                name_file           TEXT,
                transcription_whisper   TEXT,
                transcription_canary    TEXT,
                duration            FLOAT,
                start_time          FLOAT,
                score_transcription FLOAT,
                music_detection    BOOLEAN
            )
        """)
        self.conn.commit() # Commit the change

    def insert(self, segment: SegmentRecord) -> None:
        for attempt in range(5):
            try:
                self.conn.execute("""
                    INSERT INTO segments VALUES (
                        :id,
                        :name_file,
                        :transcription_whisper,
                        :transcription_canary,
                        :duration,
                        :start_time,
                        :score_transcription,
                        :music_detection
                    )
                """, segment.__dict__)

                self.conn.commit()
                return

            except sqlite3.OperationalError as e:
                logging.info(f"Database is locked, retrying... (attempt {attempt + 1}/5) erreur: {e}")
                time.sleep(0.1 * (attempt + 1))  # backoff progressif

    def get_all_files(self) -> list:
        files = self.conn.execute("""
            SELECT name_file
            FROM segments
        """).fetchall()
        files = [file[0] for file in files]
        all_name_file_process = ['_'.join(file.split("_")[:-2]) + file.split("_")[-1] for file in files]
        return list(set(all_name_file_process))
    
    def build_json_for_preprocessing(self, path_to_tar: str, name: str | None = None):
        files = self.conn.execute("""
            SELECT name_file, transcription_canary, duration
            FROM segments
            WHERE NOT music_detection AND score_transcription < 0.05
        """).fetchall()
        all_input = []
        nb_element = 0
        all_duration = 0
        for name_file, transcription, duration in files:
            single_input = {}
            single_input["speech_path"] = [path_to_tar, name_file]
            single_input["text"] = transcription
            all_input.append(single_input)
            nb_element += 1
            all_duration += duration
        print(f"Info: durée {all_duration / (60 * 60)}, nb file {nb_element}")
        if name is not None:
            if not name.endswith(".json"):
                name += ".json"
            save_json(path=name, data=all_input)

    def get_all_files_unique_transcribe(self) -> list:
        data = self.conn.execute("""
            SELECT name_file, transcription_canary
            FROM segments
        """).fetchall()
        output = {}
        files = [file[0] for file in data]
        transcription = [file[1] for file in data]
        all_name_file_process = ['_'.join(file.split("_")[:-2]) + file.split("_")[-1] for file in files]
        for file, transcription, real_name in zip(all_name_file_process, transcription, files):
            new_file = '/'.join(file.split("/")[1:])
            output[new_file] = (transcription, real_name)
        return output

