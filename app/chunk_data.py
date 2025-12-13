import csv
import os
import argparse
from typing import List, Dict, Optional
import re

class TextChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        if self.separator in text:
            chunks = self._split_by_separator(text)
        else:
            chunks = self._split_by_sentences(text)
        
        return chunks
    
    def _split_by_separator(self, text: str) -> List[str]:
        parts = text.split(self.separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if len(current_chunk) + len(part) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_large_text(part)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._apply_overlap(chunks)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'([.!?]\s+)', text)
        
        merged = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                merged.append(sentences[i] + sentences[i + 1])
            else:
                merged.append(sentences[i])
        
        chunks = []
        current_chunk = ""
        
        for sentence in merged:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._apply_overlap(chunks)
    
    def _split_large_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            break_point = end
            for i in range(end, max(start, end - 100), -1):
                if text[i] in ' \n\t' or text[i] in '.!?':
                    break_point = i + 1
                    break
            
            chunks.append(text[start:break_point].strip())
            start = break_point
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            overlapped_chunk = overlap_text + "\n\n" + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


def chunk_csv_file(
    input_file: str,
    output_file: str,
    text_column: str = "text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata_columns: Optional[List[str]] = None
):
    if metadata_columns is None:
        metadata_columns = ["url", "source", "title", "datetime", "category"]
    
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks_data = []
    
    print(f"Чтение файла: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, 1):
            text = row.get(text_column, "")
            
            if not text:
                continue
            
            text_chunks = chunker.split_text(text)
            
            for chunk_num, chunk in enumerate(text_chunks, 1):
                chunk_data = {
                    "chunk_id": f"{row_num}_{chunk_num}",
                    "original_row": row_num,
                    "chunk_number": chunk_num,
                    "total_chunks": len(text_chunks),
                    "text": chunk,
                    "chunk_size": len(chunk)
                }
                
                for col in metadata_columns:
                    chunk_data[col] = row.get(col, "")
                
                chunks_data.append(chunk_data)
            
            if row_num % 100 == 0:
                print(f"Обработано строк: {row_num}, создано чанков: {len(chunks_data)}")
    
    print(f"\nВсего обработано строк: {row_num}")
    print(f"Всего создано чанков: {len(chunks_data)}")
    
    print(f"\nСохранение в файл: {output_file}")
    fieldnames = ["chunk_id", "original_row", "chunk_number", "total_chunks", "text", "chunk_size"] + metadata_columns
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunks_data)
    
    print(f"Готово! Чанки сохранены в {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Чанкование данных из CSV файла для RAG системы"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Путь к входному CSV файлу"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Путь к выходному CSV файлу (по умолчанию: input_file_chunked.csv)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Размер чанка в символах (по умолчанию: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Размер перекрытия между чанками (по умолчанию: 200)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Название колонки с текстом (по умолчанию: text)"
    )
    
    args = parser.parse_args()
    
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_chunked.csv"
    
    if not os.path.exists(args.input_file):
        print(f"Ошибка: файл {args.input_file} не найден")
        return
    
    chunk_csv_file(
        input_file=args.input_file,
        output_file=output_file,
        text_column=args.text_column,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

if __name__ == "__main__":
    main()
