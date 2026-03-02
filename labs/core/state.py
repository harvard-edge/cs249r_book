# labs/core/state.py
# Persistent state management for the MLSys Design Ledger.
# Handles CLI (Local File) and Web (Browser LocalStorage) persistence.

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

@dataclass
class LedgerState:
    """The schema for the persistent student state."""
    track: Optional[str] = None
    current_chapter: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = ""

class DesignLedger:
    """
    The 'Save Game' manager for the MLSys curriculum.
    Ensures that decisions made in Lab 00 persist through Lab 32.
    """
    
    def __init__(self):
        self.config_dir = Path.home() / ".mlsys"
        self.file_path = self.config_dir / "ledger.json"
        self._state = LedgerState()
        self.load()

    @property
    def is_wasm(self) -> bool:
        """Detect if we are running in a browser environment (Pyodide)."""
        return sys.platform == "emscripten"

    def load(self) -> LedgerState:
        """Loads the ledger from the best available persistent storage."""
        if self.is_wasm:
            # In WASM, we rely on the browser session/localStorage.
            # Note: A full JS bridge for localStorage would be handled in the UI.
            # For now, we return the internal state.
            return self._state
        
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self._state = LedgerState(**data)
            except Exception:
                # Corrupted file? Start fresh.
                self._state = LedgerState()
        return self._state

    def save(self, track: str = None, chapter: int = None, design: dict = None):
        """Persists the design decisions to storage."""
        if track:
            self._state.track = track
        if chapter is not None:
            self._state.current_chapter = chapter
        if design:
            self._state.history.append({
                "chapter": chapter or self._state.current_chapter,
                "design": design
            })
        
        if not self.is_wasm:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(asdict(self._state), f, indent=2)
        
        # In WASM, persistence is typically handled by exporting the ledger
        # as a download button in the Marimo UI.

    def get_track(self) -> str:
        return self._state.track or "NONE"

    def get_baseline(self, chapter_id: int) -> dict:
        """
        Provides the 'Gold Standard' baseline if the student 
        hasn't completed previous labs.
        """
        # Logic to return pre-computed design for specific track/chapter
        return {}

    def __repr__(self):
        return f"DesignLedger(track={self._state.track}, ch={self._state.current_chapter})"
