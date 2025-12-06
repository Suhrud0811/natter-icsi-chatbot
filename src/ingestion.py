"""Document ingestion and indexing for ICSI MRT transcripts.

Based on ICSI corpus documentation:
- trans_guide.txt: Transcription conventions and markup
- naming.txt: Meeting/speaker ID conventions
- overview.txt: Corpus overview

This module handles:
1. XML parsing with proper structure extraction
2. Text cleaning and normalization per MRT conventions
3. Metadata extraction (speaker, timestamps, meeting type)
4. Filtering of non-content segments (digit tasks)
5. Chunking with metadata preservation
"""

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.config import (
    DATA_DIR,
    STORAGE_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NOTES_MAX_LENGTH,
)


# Meeting type codes from naming.txt
MEETING_TYPES = {
    "db": "Database issues meeting",
    "ed": "Even Deeper Understanding (NLP/AI) weekly meeting",
    "mr": "Meeting Recorder weekly meeting",
    "ns": "Network Services and Applications group meeting",
    "ro": "Robustness (signal processing) weekly meeting",
    "sr": "SRI collaboration meeting",
    "tr": "Meeting Recorder transcriber's meeting",
    "uw": "UW collaboration meeting",
}


@dataclass
class Utterance:
    """A single utterance from a meeting transcript."""
    speaker: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    is_digit_task: bool = False


@dataclass 
class MeetingMetadata:
    """Metadata extracted from MRT file."""
    meeting_id: str
    session: str
    date_time: Optional[str] = None
    meeting_type: Optional[str] = None
    meeting_type_desc: Optional[str] = None
    notes: Optional[str] = None
    participants: Dict[str, str] = field(default_factory=dict)
    duration_seconds: Optional[float] = None


def parse_meeting_id(meeting_id: str) -> tuple:
    """Parse meeting ID into components.
    
    Format: Xyz### where:
    - X = location (B = Berkeley/ICSI)
    - yz = meeting type code
    - ### = meeting number
    
    Example: Bmr001 -> (B, mr, 001)
    """
    if len(meeting_id) >= 6:
        location = meeting_id[0]
        meeting_type = meeting_id[1:3]
        number = meeting_id[3:]
        return location, meeting_type, number
    return None, None, None


def parse_speaker_id(speaker_id: str) -> dict:
    """Parse speaker ID into components.
    
    Format: XY### where:
    - X = m/f/u/x (male/female/unknown/computer)
    - Y = e/n (native/non-native English)
    - ### = unique number
    
    Example: me011 -> {gender: male, native: True, id: 011}
    """
    info = {"raw_id": speaker_id}
    if len(speaker_id) >= 5:
        gender_map = {"m": "male", "f": "female", "u": "unknown", "x": "computer"}
        info["gender"] = gender_map.get(speaker_id[0], "unknown")
        info["native_english"] = speaker_id[1] == "e"
        info["speaker_num"] = speaker_id[2:]
    return info


def clean_text(text: str) -> str:
    """Clean and normalize transcript text.
    
    Based on MRT conventions from trans_guide.txt:
    - Preserve meaningful markers in readable form
    - Remove XML artifacts
    - Normalize whitespace
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Keep vocal sounds as context clues
    vocal_sound_pattern = r'<VocalSound\s+Description="([^"]+)"\s*/>'
    text = re.sub(vocal_sound_pattern, r'[\1]', text)
    
    non_vocal_sound_pattern = r'<NonVocalSound\s+Description="([^"]+)"\s*/>'
    text = re.sub(non_vocal_sound_pattern, r'[\1]', text)
    
    # Convert pauses to readable format
    text = re.sub(r'<Pause\s*/>', '...', text)
    
    # Handle emphasis - keep the word, note emphasis
    emphasis_pattern = r'<Emphasis>\s*([^<]+)\s*</Emphasis>'
    text = re.sub(emphasis_pattern, r'\1', text)
    
    # Handle uncertain transcriptions - keep with marker
    uncertain_pattern = r'<Uncertain>\s*([^<]+)\s*</Uncertain>'
    text = re.sub(uncertain_pattern, r'(\1?)', text)
    
    uncertain_unintelligible = r'<Uncertain[^>]*>\s*@@\s*</Uncertain>'
    text = re.sub(uncertain_unintelligible, '(unintelligible)', text)
    
    # Handle foreign words
    foreign_pattern = r'<Foreign[^>]*>\s*([^<]+)\s*</Foreign>'
    text = re.sub(foreign_pattern, r'\1', text)
    
    # Handle pronunciation notes - just keep the word
    pronounce_pattern = r'<Pronounce[^>]*>\s*([^<]+)\s*</Pronounce>'
    text = re.sub(pronounce_pattern, r'\1', text)
    
    # Remove comment tags but could optionally keep description
    comment_pattern = r'<Comment\s+Description="([^"]+)"\s*/>'
    text = re.sub(comment_pattern, '', text)
    
    # Remove any remaining XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up O_K -> OK (common in transcripts)
    text = re.sub(r'\bO_K\b', 'OK', text)
    
    # Clean up P_D_A -> PDA, etc. (underscore notation for acronyms)
    three_letter_acronym = r'(\w)_(\w)_(\w)\b'
    text = re.sub(three_letter_acronym, r'\1\2\3', text)
    
    two_letter_acronym = r'(\w)_(\w)\b'
    text = re.sub(two_letter_acronym, r'\1\2', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def is_empty_or_noise(text: str) -> bool:
    """Check if text is empty or just noise/backchannels."""
    if not text:
        return True
    
    # Very short utterances that are just vocal sounds
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True
    
    # Just brackets (vocal sounds only)
    if re.match(r'^\[[^\]]+\]$', cleaned):
        return True
    
    return False


def extract_preamble_info(preamble: ET.Element) -> tuple:
    """Extract notes and participant info from Preamble."""
    notes = None
    participants = {}
    
    # Get notes
    notes_elem = preamble.find("Notes")
    if notes_elem is not None and notes_elem.text:
        notes = notes_elem.text.strip()
    
    # Get participants
    participants_elem = preamble.find("Participants")
    if participants_elem is not None:
        for participant in participants_elem.findall("Participant"):
            name = participant.get("Name")
            channel = participant.get("Channel")
            if name:
                participants[name] = channel
    
    return notes, participants


def parse_mrt_file(file_path: Path) -> Optional[Document]:
    """Parse a single MRT (Meeting Room Transcript) XML file.
    
    Extracts:
    - Meeting metadata (session, date, type, participants)
    - Utterances with speaker labels and timestamps
    - Filters out digit task segments
    
    Returns a Document with cleaned text and rich metadata.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract meeting-level metadata
        meeting_id = file_path.stem
        session = root.get("Session", meeting_id)
        date_time = root.get("DateTimeStamp")
        
        # Parse meeting type from ID
        _, type_code, _ = parse_meeting_id(meeting_id)
        meeting_type_desc = MEETING_TYPES.get(type_code, "Unknown meeting type")
        
        # Extract preamble info
        notes = None
        participants = {}
        preamble = root.find("Preamble")
        if preamble is not None:
            notes, participants = extract_preamble_info(preamble)
        
        # Extract transcript segments
        transcript = root.find("Transcript")
        if transcript is None:
            return None
        
        utterances: List[Utterance] = []
        speakers: Set[str] = set()
        
        for segment in transcript.findall("Segment"):
            # Check if this is a digit task segment - SKIP these
            is_digit_task = segment.get("DigitTask") == "true"
            if is_digit_task:
                continue
            
            speaker = segment.get("Participant", "Unknown")
            start_time = segment.get("StartTime")
            end_time = segment.get("EndTime")
            
            # Get segment as XML string to process markers
            segment_xml = ET.tostring(segment, encoding='unicode')
            cleaned_text = clean_text(segment_xml)
            
            if cleaned_text and not is_empty_or_noise(cleaned_text):
                utterances.append(Utterance(
                    speaker=speaker,
                    text=cleaned_text,
                    start_time=float(start_time) if start_time else None,
                    end_time=float(end_time) if end_time else None,
                    is_digit_task=False,
                ))
                speakers.add(speaker)
        
        if not utterances:
            return None
        
        # Format utterances with speaker labels
        formatted_lines = []
        for utt in utterances:
            # Use speaker ID directly (they're standardized: me011, fn002, etc.)
            formatted_lines.append(f"[{utt.speaker}]: {utt.text}")
        
        full_text = "\n".join(formatted_lines)
        
        # Calculate duration
        start_times = [u.start_time for u in utterances if u.start_time is not None]
        end_times = [u.end_time for u in utterances if u.end_time is not None]
        duration = None
        if start_times and end_times:
            duration = max(end_times) - min(start_times)
        
        # Build comprehensive metadata
        metadata = {
            "meeting_id": meeting_id,
            "session": session,
            "source": str(file_path),
            "num_utterances": len(utterances),
            "speakers": list(speakers),
            "num_speakers": len(speakers),
            "meeting_type": type_code,
            "meeting_type_description": meeting_type_desc,
        }
        
        if date_time:
            metadata["date_time"] = date_time
        if notes:
            metadata["notes"] = notes[:NOTES_MAX_LENGTH]  # Truncate long notes
        if participants:
            metadata["participants"] = participants
        if start_times:
            metadata["start_time"] = min(start_times)
        if end_times:
            metadata["end_time"] = max(end_times)
        if duration:
            metadata["duration_seconds"] = duration
        
        return Document(text=full_text, metadata=metadata)
        
    except ET.ParseError as e:
        print(f"Warning: XML parse error in {file_path}: {e}")
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")
    
    return None


def load_transcripts(data_dir: Path = DATA_DIR) -> List[Document]:
    """Load all MRT transcript files from the data directory.
    
    Returns a list of Documents with cleaned text and metadata.
    Skips preambles.mrt (contains only preamble templates).
    """
    documents = []
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please download the ICSI corpus and place .mrt files in data/transcripts/"
        )
    
    mrt_files = list(data_dir.glob("*.mrt"))
    
    # Filter out non-transcript files
    mrt_files = [f for f in mrt_files if f.stem != "preambles"]
    
    if not mrt_files:
        raise FileNotFoundError(
            f"No .mrt files found in {data_dir}\n"
            "Please download the ICSI corpus from:\n"
            "https://groups.inf.ed.ac.uk/ami/icsi/download/"
        )
    
    print(f"Loading {len(mrt_files)} transcript files...")
    
    total_utterances = 0
    all_speakers = set()
    meeting_types = {}
    
    for file_path in sorted(mrt_files):
        doc = parse_mrt_file(file_path)
        if doc:
            documents.append(doc)
            total_utterances += doc.metadata.get("num_utterances", 0)
            all_speakers.update(doc.metadata.get("speakers", []))
            
            # Count meeting types
            mt = doc.metadata.get("meeting_type", "unknown")
            meeting_types[mt] = meeting_types.get(mt, 0) + 1
    
    print(f"Successfully loaded {len(documents)} transcripts")
    print(f"Total utterances: {total_utterances} (excluding digit tasks)")
    print(f"Unique speakers: {len(all_speakers)}")
    print(f"Meeting types: {meeting_types}")
    
    return documents


def create_index(
    documents: List[Document],
    persist: bool = True,
) -> VectorStoreIndex:
    """Create a vector store index from documents.
    
    Uses sentence-based chunking (300-800 tokens) optimized for 
    conversational transcripts. Metadata is preserved on each chunk.
    """
    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    
    # Initialize LLM
    llm = OpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
    )
    
    # Create node parser with sentence-aware chunking
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    print("Creating vector index...")
    print(f"Chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP} tokens")
    
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[node_parser],
    )
    
    if persist:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        print(f"Index persisted to {STORAGE_DIR}")
    
    return index


def load_or_create_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Load existing index or create a new one.
    
    Args:
        force_rebuild: If True, rebuild index even if storage exists
    """
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    
    # Try to load existing index
    if not force_rebuild and STORAGE_DIR.exists():
        try:
            print("Loading existing index...")
            storage_context = StorageContext.from_defaults(
                persist_dir=str(STORAGE_DIR)
            )
            index = load_index_from_storage(
                storage_context,
                embed_model=embed_model,
            )
            print("Index loaded successfully")
            return index
        except Exception as e:
            print(f"Could not load existing index: {e}")
            print("Creating new index...")
    
    # Create new index
    documents = load_transcripts()
    return create_index(documents)
