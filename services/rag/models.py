from dataclasses import dataclass

@dataclass
class RAGChunk:
    text: str
    score: float