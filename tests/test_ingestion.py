"""Tests for document ingestion module."""

from pathlib import Path

import pytest

from src.ingestion import (
    parse_mrt_file,
    load_transcripts,
    clean_text,
    is_empty_or_noise,
    parse_meeting_id,
    parse_speaker_id,
    MEETING_TYPES,
)


class TestParseMeetingId:
    """Tests for meeting ID parsing."""
    
    def test_parse_standard_meeting_id(self):
        """Test parsing standard meeting ID like Bmr001."""
        location, type_code, number = parse_meeting_id("Bmr001")
        assert location == "B"
        assert type_code == "mr"
        assert number == "001"
    
    def test_parse_different_meeting_types(self):
        """Test parsing different meeting type codes."""
        _, type_code, _ = parse_meeting_id("Bed005")
        assert type_code == "ed"
        
        _, type_code, _ = parse_meeting_id("Bro017")
        assert type_code == "ro"
    
    def test_parse_short_id_returns_none(self):
        """Test that short IDs return None."""
        location, type_code, number = parse_meeting_id("Bmr")
        assert location is None


class TestParseSpeakerId:
    """Tests for speaker ID parsing."""
    
    def test_parse_male_native_speaker(self):
        """Test parsing male native English speaker ID."""
        info = parse_speaker_id("me011")
        assert info["gender"] == "male"
        assert info["native_english"] is True
        assert info["speaker_num"] == "011"
    
    def test_parse_female_nonnative_speaker(self):
        """Test parsing female non-native speaker ID."""
        info = parse_speaker_id("fn002")
        assert info["gender"] == "female"
        assert info["native_english"] is False
    
    def test_parse_unknown_speaker(self):
        """Test parsing unknown speaker ID."""
        info = parse_speaker_id("ue001")
        assert info["gender"] == "unknown"


class TestCleanText:
    """Tests for text cleaning and normalization."""
    
    def test_clean_empty_text(self):
        """Test cleaning empty string."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_clean_html_entities(self):
        """Test HTML entity decoding."""
        assert "&" in clean_text("Tom &amp; Jerry")
    
    def test_clean_vocal_sounds(self):
        """Test vocal sound markers are converted."""
        result = clean_text('<VocalSound Description="laugh"/>')
        assert "[laugh]" in result
    
    def test_clean_pauses(self):
        """Test pause markers are converted."""
        result = clean_text('Hello <Pause/> world')
        assert "..." in result
        assert "<Pause" not in result
    
    def test_clean_emphasis(self):
        """Test emphasis markers are handled."""
        result = clean_text('<Emphasis> important </Emphasis>')
        assert "important" in result
        assert "<Emphasis>" not in result
    
    def test_clean_uncertain(self):
        """Test uncertain transcriptions are marked."""
        result = clean_text('<Uncertain> maybe </Uncertain>')
        assert "maybe" in result
        assert "?" in result
    
    def test_clean_ok_underscore(self):
        """Test O_K is converted to OK."""
        result = clean_text("O_K, let's start")
        assert "OK" in result
        assert "O_K" not in result
    
    def test_clean_acronyms(self):
        """Test acronym underscores are removed."""
        result = clean_text("the P_D_A device")
        assert "PDA" in result
        assert "P_D_A" not in result
    
    def test_clean_whitespace_normalization(self):
        """Test whitespace is normalized."""
        result = clean_text("Hello    world\n\ttab")
        assert "  " not in result


class TestIsEmptyOrNoise:
    """Tests for empty/noise detection."""
    
    def test_empty_is_noise(self):
        """Empty text is considered noise."""
        assert is_empty_or_noise("")
        assert is_empty_or_noise(None)
    
    def test_very_short_is_noise(self):
        """Very short text is noise."""
        assert is_empty_or_noise("a")
    
    def test_just_vocal_sound_is_noise(self):
        """Just a vocal sound marker is noise."""
        assert is_empty_or_noise("[laugh]")
        assert is_empty_or_noise("[breath]")
    
    def test_real_content_not_noise(self):
        """Real content should not be flagged."""
        assert not is_empty_or_noise("We discussed the project")
        assert not is_empty_or_noise("OK, sounds good")


class TestParseMrtFile:
    """Tests for MRT file parsing."""
    
    def test_parse_valid_mrt_file(self, tmp_path):
        """Test parsing a valid MRT file."""
        mrt_content = """<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Bmr001" DateTimeStamp="2000-02-02-1700">
  <Preamble>
    <Notes>Test meeting notes</Notes>
    <Participants>
      <Participant Name="me011" Channel="chan1"/>
      <Participant Name="me013" Channel="chan0"/>
    </Participants>
  </Preamble>
  <Transcript StartTime="0.0" EndTime="100.0">
    <Segment StartTime="2.0" EndTime="4.0" Participant="me011">
      O_K, so we are live.
    </Segment>
    <Segment StartTime="4.0" EndTime="6.0" Participant="me013">
      Let us begin the meeting.
    </Segment>
  </Transcript>
</Meeting>
"""
        mrt_file = tmp_path / "Bmr001.mrt"
        mrt_file.write_text(mrt_content)
        
        doc = parse_mrt_file(mrt_file)
        
        assert doc is not None
        assert doc.metadata["meeting_id"] == "Bmr001"
        assert doc.metadata["session"] == "Bmr001"
        assert doc.metadata["meeting_type"] == "mr"
        assert doc.metadata["num_utterances"] == 2
        assert doc.metadata["num_speakers"] == 2
        assert "OK" in doc.text
    
    def test_parse_filters_digit_tasks(self, tmp_path):
        """Test that digit task segments are filtered out."""
        mrt_content = """<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Bmr001">
  <Transcript StartTime="0.0" EndTime="100.0">
    <Segment StartTime="2.0" EndTime="4.0" Participant="me011">
      Regular meeting content here.
    </Segment>
    <Segment StartTime="10.0" EndTime="12.0" Participant="me011" DigitTask="true">
      one two three four five
    </Segment>
    <Segment StartTime="20.0" EndTime="22.0" Participant="me013">
      Back to regular discussion.
    </Segment>
  </Transcript>
</Meeting>
"""
        mrt_file = tmp_path / "Bmr002.mrt"
        mrt_file.write_text(mrt_content)
        
        doc = parse_mrt_file(mrt_file)
        
        assert doc is not None
        assert doc.metadata["num_utterances"] == 2
        assert "one two three" not in doc.text
        assert "Regular meeting content" in doc.text
    
    def test_parse_extracts_metadata(self, tmp_path):
        """Test that metadata is properly extracted."""
        mrt_content = """<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Bed005" DateTimeStamp="2001-03-15-1400">
  <Preamble>
    <Notes>Important technical notes here.</Notes>
    <Participants>
      <Participant Name="fn002" Channel="chan0"/>
    </Participants>
  </Preamble>
  <Transcript StartTime="0.0" EndTime="3600.0">
    <Segment StartTime="10.0" EndTime="20.0" Participant="fn002">
      Let us discuss NLP topics.
    </Segment>
  </Transcript>
</Meeting>
"""
        mrt_file = tmp_path / "Bed005.mrt"
        mrt_file.write_text(mrt_content)
        
        doc = parse_mrt_file(mrt_file)
        
        assert doc is not None
        assert doc.metadata["meeting_type"] == "ed"
        assert "Even Deeper Understanding" in doc.metadata["meeting_type_description"]
        assert doc.metadata["date_time"] == "2001-03-15-1400"
        assert "fn002" in doc.metadata["participants"]
    
    def test_parse_handles_vocal_sounds(self, tmp_path):
        """Test that vocal sounds are properly converted."""
        mrt_content = """<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Bmr003">
  <Transcript StartTime="0.0" EndTime="100.0">
    <Segment StartTime="5.0" EndTime="7.0" Participant="me011">
      That is funny <VocalSound Description="laugh"/>
    </Segment>
  </Transcript>
</Meeting>
"""
        mrt_file = tmp_path / "Bmr003.mrt"
        mrt_file.write_text(mrt_content)
        
        doc = parse_mrt_file(mrt_file)
        
        assert doc is not None
        assert "[laugh]" in doc.text
    
    def test_parse_empty_transcript(self, tmp_path):
        """Test parsing file with empty transcript."""
        mrt_content = """<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="Bmr099">
  <Transcript StartTime="0.0" EndTime="0.0">
  </Transcript>
</Meeting>
"""
        mrt_file = tmp_path / "Bmr099.mrt"
        mrt_file.write_text(mrt_content)
        
        doc = parse_mrt_file(mrt_file)
        
        assert doc is None
    
    def test_parse_invalid_xml(self, tmp_path):
        """Test parsing an invalid XML file."""
        mrt_file = tmp_path / "invalid.mrt"
        mrt_file.write_text("This is not valid XML")
        
        doc = parse_mrt_file(mrt_file)
        
        assert doc is None


class TestLoadTranscripts:
    """Tests for transcript loading."""
    
    def test_load_transcripts_no_directory(self):
        """Test loading from non-existent directory."""
        fake_path = Path("/nonexistent/path")
        
        with pytest.raises(FileNotFoundError):
            load_transcripts(fake_path)
    
    def test_load_transcripts_empty_directory(self, tmp_path):
        """Test loading from directory with no MRT files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_transcripts(tmp_path)
        
        assert "No .mrt files found" in str(exc_info.value)
    
    def test_load_transcripts_success(self, tmp_path):
        """Test successfully loading multiple MRT files."""
        for i, meeting_type in enumerate(["Bmr", "Bed", "Bro"]):
            mrt_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Meeting Session="{meeting_type}00{i+1}">
  <Transcript StartTime="0.0" EndTime="100.0">
    <Segment StartTime="2.0" EndTime="4.0" Participant="me011">
      Content of meeting {i+1}
    </Segment>
  </Transcript>
</Meeting>
"""
            (tmp_path / f"{meeting_type}00{i+1}.mrt").write_text(mrt_content)
        
        docs = load_transcripts(tmp_path)
        
        assert len(docs) == 3
        meeting_types = [d.metadata.get("meeting_type") for d in docs]
        assert "mr" in meeting_types
        assert "ed" in meeting_types
        assert "ro" in meeting_types


class TestMeetingTypes:
    """Tests for meeting type constants."""
    
    def test_all_meeting_types_defined(self):
        """Test that all expected meeting types are defined."""
        expected_types = ["db", "ed", "mr", "ns", "ro", "sr", "tr", "uw"]
        for mt in expected_types:
            assert mt in MEETING_TYPES
    
    def test_meeting_type_descriptions_not_empty(self):
        """Test that all meeting types have descriptions."""
        for mt, desc in MEETING_TYPES.items():
            assert desc, f"Meeting type {mt} has empty description"
