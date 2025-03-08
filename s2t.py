import os
import time
import whisper
import pandas as pd
import numpy as np
from jiwer import wer, mer, wil
from tqdm import tqdm
import torch
import json
import wave
import re
import string

# Try importing transformers for Wav2Vec2
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import librosa

    wav2vec2_available = True
except ImportError:
    print("Transformers or librosa not installed. Install with:")
    print("pip install transformers librosa")
    wav2vec2_available = False

# Try importing vosk
try:
    from vosk import Model, KaldiRecognizer

    vosk_available = True
except ImportError:
    print("Vosk not installed. Install with:")
    print("pip install vosk")
    vosk_available = False


def test_whisper(audio_file, model):
    """Run Whisper transcription on an audio file."""
    start_time = time.time()
    # Force English language and transcription mode (not translation)
    result = model.transcribe(audio_file, language="en", task="transcribe")
    processing_time = time.time() - start_time
    return result["text"].strip(), processing_time


def test_wav2vec2(audio_file, processor, model):
    """Run Wav2Vec2 transcription on an audio file."""
    start_time = time.time()

    # Load audio
    speech_array, sampling_rate = librosa.load(audio_file, sr=16000)

    # Tokenize
    input_values = processor(
        speech_array, sampling_rate=16000, return_tensors="pt"
    ).input_values

    # Take logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    processing_time = time.time() - start_time
    return transcription, processing_time


def test_vosk(audio_file, model):
    """Run Vosk transcription on an audio file."""
    start_time = time.time()
    import tempfile  # Import tempfile at the function level

    print(f"Attempting Vosk transcription for {audio_file}")

    try:
        # Make sure we have soundfile installed
        import soundfile as sf
    except ImportError:
        print("soundfile not installed. Install with: pip install soundfile")
        return "ERROR: Missing soundfile dependency", time.time() - start_time

    try:
        # Convert to wav for Vosk if needed
        if not audio_file.endswith(".wav"):
            print(f"Converting {audio_file} to WAV format for Vosk")
            # Use librosa to load and resample if not a wav file
            speech_array, sr = librosa.load(audio_file, sr=16000)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name
            sf.write(temp_wav, speech_array, sr, subtype="PCM_16")
            audio_file = temp_wav
            print(f"Converted to temporary WAV file: {audio_file}")

        # Open the WAV file
        wf = wave.open(audio_file, "rb")

        # Check if the wavefile is in the correct format for Vosk
        sample_rate = wf.getframerate()
        print(
            f"WAV format: Channels={wf.getnchannels()}, Width={wf.getsampwidth()}, Rate={sample_rate}"
        )
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            print(f"Audio file {audio_file} must be mono PCM WAV format for Vosk")
            print("Converting audio to correct format...")
            wf.close()

            # Re-convert with explicit parameters for mono, 16-bit PCM
            speech_array, sr = librosa.load(audio_file, sr=16000, mono=True)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name
            sf.write(temp_wav, speech_array, 16000, subtype="PCM_16", format="WAV")

            # Remove the old temporary file if it's different
            if audio_file != temp_wav and audio_file.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(audio_file)
                except:
                    pass

            audio_file = temp_wav
            wf = wave.open(audio_file, "rb")
            sample_rate = wf.getframerate()
            print(
                f"Converted WAV format: Channels={wf.getnchannels()}, Width={wf.getsampwidth()}, Rate={sample_rate}"
            )

        # Create recognizer
        print(f"Initializing Vosk recognizer with sample rate: {sample_rate}")
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)

        # Process the entire file
        transcription = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "text" in result:
                    transcription += result["text"] + " "

        # Get final result
        final_result = json.loads(rec.FinalResult())
        if "text" in final_result:
            transcription += final_result["text"]

        # Clean up temp file if created
        if audio_file.startswith(tempfile.gettempdir()):
            try:
                os.unlink(audio_file)
            except:
                pass

        wf.close()
        processing_time = time.time() - start_time
        print(
            f"Vosk transcription completed in {processing_time:.2f}s: {transcription[:50]}..."
        )
        return transcription.strip(), processing_time

    except Exception as e:
        import traceback

        print(f"Error in Vosk transcription: {str(e)}")
        print(traceback.format_exc())
        return f"ERROR: {str(e)}", time.time() - start_time


def normalize_text(text):
    """
    Normalize text for fair comparison by:
    - Converting to lowercase
    - Removing punctuation
    - Standardizing spacing
    - Standardizing numbers
    """
    if text is None:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace common non-standard representations
    text = text.replace("'s", " s")  # Possessives
    text = text.replace("'", "")  # Apostrophes
    text = text.replace("n't", " not")  # Contractions

    # Remove punctuation (except hyphens in compound words)
    text = re.sub(r"[^\w\s-]", "", text)

    # Replace hyphens with spaces
    text = text.replace("-", " ")

    # Standardize whitespace
    text = " ".join(text.split())

    return text


def calculate_error_metrics(reference, hypothesis):
    """Calculate WER, MER, and WIL with normalized text for fair comparison."""
    # Normalize both texts
    norm_reference = normalize_text(reference)
    norm_hypothesis = normalize_text(hypothesis)

    # Calculate metrics
    try:
        error_wer = wer(norm_reference, norm_hypothesis)
        error_mer = mer(norm_reference, norm_hypothesis)
        error_wil = wil(norm_reference, norm_hypothesis)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print(f"Reference: '{norm_reference}'")
        print(f"Hypothesis: '{norm_hypothesis}'")
        return 1.0, 1.0, 1.0

    return error_wer, error_mer, error_wil


def compare_speech_recognition_models(
    audio_dir,
    reference_path,
    whisper_model_name="base",
    vosk_model_path="vosk-model-en-us-0.22",
):
    """
    Compare Whisper, Wav2Vec2, and Vosk on the same audio files.

    Parameters:
    audio_dir (str): Directory containing audio files
    reference_path (str): Path to CSV file with reference transcripts
    whisper_model_name (str): Whisper model size (tiny, base, small, medium, large)
    vosk_model_path (str): Path to the Vosk model directory

    Returns:
    pd.DataFrame: Comparative results
    """
    # Load reference transcripts
    print("Loading reference transcripts")
    references = pd.read_csv(reference_path)

    # Load Whisper model
    print(f"Loading Whisper model: {whisper_model_name}")
    whisper_model = whisper.load_model(whisper_model_name)

    # Load Wav2Vec2 model if available
    wav2vec2_processor = None
    wav2vec2_model = None
    if wav2vec2_available:
        try:
            print("Loading Wav2Vec2 model (this may take a minute)...")
            model_name = "facebook/wav2vec2-base-960h"
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
            wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            print("Wav2Vec2 model loaded successfully")
        except Exception as e:
            print(f"Error loading Wav2Vec2 model: {e}")
            wav2vec2_processor = None
            wav2vec2_model = None

    # Load Vosk model if available
    vosk_model = None
    if vosk_available:
        try:
            print(f"Loading Vosk model from: {vosk_model_path}")
            if not os.path.exists(vosk_model_path):
                print(f"Warning: Vosk model path {vosk_model_path} does not exist.")
                print("Download a model from https://alphacephei.com/vosk/models")
                print("and extract it to the specified path.")
            else:
                # Print model structure to debug
                model_files = os.listdir(vosk_model_path)
                print(f"Vosk model directory contains: {model_files}")

                # Check if we're using the graph model
                is_graph_model = (
                    "graph" in vosk_model_path.lower()
                    or "lgraph" in vosk_model_path.lower()
                )
                if is_graph_model:
                    print("Detected graph-based Vosk model")

                # Load the model
                vosk_model = Model(vosk_model_path)
                print("Vosk model loaded successfully")
        except Exception as e:
            import traceback

            print(f"Error loading Vosk model: {e}")
            print(traceback.format_exc())
            vosk_model = None

    # Initialize results dictionary
    results = {
        "filename": [],
        "reference_text": [],
        "whisper_text": [],
        "whisper_wer": [],
        "whisper_mer": [],
        "whisper_wil": [],
        "whisper_time": [],
    }

    # Add Wav2Vec2 columns if available
    if wav2vec2_model is not None:
        results.update(
            {
                "wav2vec2_text": [],
                "wav2vec2_wer": [],
                "wav2vec2_mer": [],
                "wav2vec2_wil": [],
                "wav2vec2_time": [],
            }
        )

    # Add Vosk columns if available
    if vosk_model is not None:
        results.update(
            {
                "vosk_text": [],
                "vosk_wer": [],
                "vosk_mer": [],
                "vosk_wil": [],
                "vosk_time": [],
            }
        )

    # Get all audio files in the directory
    audio_files = [
        f
        for f in os.listdir(audio_dir)
        if f.endswith((".wav"))  # Only process WAV files
    ]

    if not audio_files:
        print(
            f"No WAV files found in {audio_dir}. Please ensure your converted WAV files are in this directory."
        )
        return pd.DataFrame()

    # Process each audio file
    print(f"Processing {len(audio_files)} audio files...")
    for filename in tqdm(audio_files):
        file_path = os.path.join(audio_dir, filename)
        file_id = os.path.splitext(filename)[0]

        # Find reference transcript for this file
        if file_id not in references["file_id"].values:
            print(f"Warning: No reference transcript found for {file_id}, skipping")
            continue

        reference_text = references.loc[
            references["file_id"] == file_id, "transcript"
        ].values[0]

        # Add file and reference to results
        results["filename"].append(filename)
        results["reference_text"].append(reference_text)

        # Test with Whisper
        try:
            whisper_text, whisper_time = test_whisper(file_path, whisper_model)

            # Calculate error metrics with fair comparison
            whisper_wer_val, whisper_mer_val, whisper_wil_val = calculate_error_metrics(
                reference_text, whisper_text
            )

            # Add to results
            results["whisper_text"].append(whisper_text)
            results["whisper_wer"].append(whisper_wer_val)
            results["whisper_mer"].append(whisper_mer_val)
            results["whisper_wil"].append(whisper_wil_val)
            results["whisper_time"].append(whisper_time)
        except Exception as e:
            print(f"Error with Whisper on file {filename}: {str(e)}")
            results["whisper_text"].append("ERROR")
            results["whisper_wer"].append(None)
            results["whisper_mer"].append(None)
            results["whisper_wil"].append(None)
            results["whisper_time"].append(None)

        # Test with Wav2Vec2 if available
        if wav2vec2_model is not None and wav2vec2_processor is not None:
            try:
                wav2vec2_text, wav2vec2_time = test_wav2vec2(
                    file_path, wav2vec2_processor, wav2vec2_model
                )

                # Calculate error metrics with fair comparison
                wav2vec2_wer_val, wav2vec2_mer_val, wav2vec2_wil_val = (
                    calculate_error_metrics(reference_text, wav2vec2_text)
                )

                # Add to results
                results["wav2vec2_text"].append(wav2vec2_text)
                results["wav2vec2_wer"].append(wav2vec2_wer_val)
                results["wav2vec2_mer"].append(wav2vec2_mer_val)
                results["wav2vec2_wil"].append(wav2vec2_wil_val)
                results["wav2vec2_time"].append(wav2vec2_time)
            except Exception as e:
                print(f"Error with Wav2Vec2 on file {filename}: {str(e)}")
                results["wav2vec2_text"].append("ERROR")
                results["wav2vec2_wer"].append(None)
                results["wav2vec2_mer"].append(None)
                results["wav2vec2_wil"].append(None)
                results["wav2vec2_time"].append(None)

        # Test with Vosk if available
        if vosk_model is not None:
            try:
                vosk_text, vosk_time = test_vosk(file_path, vosk_model)

                # Calculate error metrics with fair comparison
                vosk_wer_val, vosk_mer_val, vosk_wil_val = calculate_error_metrics(
                    reference_text, vosk_text
                )

                # Add to results
                results["vosk_text"].append(vosk_text)
                results["vosk_wer"].append(vosk_wer_val)
                results["vosk_mer"].append(vosk_mer_val)
                results["vosk_wil"].append(vosk_wil_val)
                results["vosk_time"].append(vosk_time)
            except Exception as e:
                print(f"Error with Vosk on file {filename}: {str(e)}")
                results["vosk_text"].append("ERROR")
                results["vosk_wer"].append(None)
                results["vosk_mer"].append(None)
                results["vosk_wil"].append(None)
                results["vosk_time"].append(None)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary metrics
    print("\nSummary Metrics (Normalized for Fair Comparison):")

    print("\nWhisper Results:")
    print(f"Word Error Rate (WER): {results_df['whisper_wer'].mean():.4f}")
    print(f"Match Error Rate (MER): {results_df['whisper_mer'].mean():.4f}")
    print(f"Word Information Lost (WIL): {results_df['whisper_wil'].mean():.4f}")
    print(f"Average Processing Time: {results_df['whisper_time'].mean():.2f} seconds")

    if wav2vec2_model is not None:
        print("\nWav2Vec2 Results:")
        print(f"Word Error Rate (WER): {results_df['wav2vec2_wer'].mean():.4f}")
        print(f"Match Error Rate (MER): {results_df['wav2vec2_mer'].mean():.4f}")
        print(f"Word Information Lost (WIL): {results_df['wav2vec2_wil'].mean():.4f}")
        print(
            f"Average Processing Time: {results_df['wav2vec2_time'].mean():.2f} seconds"
        )

    if vosk_model is not None:
        print("\nVosk Results:")
        print(f"Word Error Rate (WER): {results_df['vosk_wer'].mean():.4f}")
        print(f"Match Error Rate (MER): {results_df['vosk_mer'].mean():.4f}")
        print(f"Word Information Lost (WIL): {results_df['vosk_wil'].mean():.4f}")
        print(f"Average Processing Time: {results_df['vosk_time'].mean():.2f} seconds")

    return results_df


def create_pivot_table(results_df):
    """Create a pivot table with the reference as a row and models as columns."""
    # Create a list to store the pivoted data
    pivot_data = []

    # Process each file
    for filename in results_df["filename"].unique():
        file_data = results_df[results_df["filename"] == filename]

        # Get reference text
        reference_text = file_data["reference_text"].iloc[0]

        # Create a row for reference
        ref_row = {
            "filename": filename,
            "metric_type": "reference",
            "whisper": reference_text,
            "wav2vec2": reference_text,
            "vosk": reference_text,
        }
        pivot_data.append(ref_row)

        # Create rows for each metric
        metric_types = ["text", "wer", "mer", "wil", "time"]

        for metric_type in metric_types:
            row = {"filename": filename, "metric_type": metric_type}

            # Get whisper value
            whisper_col = f"whisper_{metric_type}"
            if whisper_col in file_data.columns:
                row["whisper"] = file_data[whisper_col].iloc[0]

            # Get wav2vec2 value
            wav2vec2_col = f"wav2vec2_{metric_type}"
            if wav2vec2_col in file_data.columns:
                row["wav2vec2"] = file_data[wav2vec2_col].iloc[0]

            # Get vosk value
            vosk_col = f"vosk_{metric_type}"
            if vosk_col in file_data.columns:
                row["vosk"] = file_data[vosk_col].iloc[0]

            pivot_data.append(row)

    # Convert to DataFrame
    pivot_df = pd.DataFrame(pivot_data)

    return pivot_df


def print_readable_table(pivot_df):
    """Print the pivot table in a readable format with reference as a row."""
    # Group by filename
    for filename in pivot_df["filename"].unique():
        file_df = pivot_df[pivot_df["filename"] == filename]

        # Sort by metric type with reference first, then text, then metrics
        metric_order = {
            "reference": 0,
            "text": 1,
            "wer": 2,
            "mer": 3,
            "wil": 4,
            "time": 5,
        }
        file_df = file_df.sort_values(
            by="metric_type", key=lambda x: x.map(metric_order)
        )

        # Print the table for this file
        print(f"\nFile: {filename}")
        print("-" * 100)

        # Print header
        print(f"{'metric_type':<12} {'whisper':<25} {'wav2vec2':<25} {'vosk':<25}")
        print("-" * 100)

        # Print rows
        for _, row in file_df.iterrows():
            metric = row["metric_type"]
            whisper_val = row.get("whisper", "N/A")
            wav2vec2_val = row.get("wav2vec2", "N/A")
            vosk_val = row.get("vosk", "N/A")

            # Format numerical values
            if metric in ["wer", "mer", "wil"]:
                if isinstance(whisper_val, (int, float)) and not np.isnan(whisper_val):
                    whisper_val = f"{whisper_val:.4f}"
                if isinstance(wav2vec2_val, (int, float)) and not np.isnan(
                    wav2vec2_val
                ):
                    wav2vec2_val = f"{wav2vec2_val:.4f}"
                if isinstance(vosk_val, (int, float)) and not np.isnan(vosk_val):
                    vosk_val = f"{vosk_val:.4f}"
            elif metric == "time":
                if isinstance(whisper_val, (int, float)) and not np.isnan(whisper_val):
                    whisper_val = f"{whisper_val:.4f}s"
                if isinstance(wav2vec2_val, (int, float)) and not np.isnan(
                    wav2vec2_val
                ):
                    wav2vec2_val = f"{wav2vec2_val:.4f}s"
                if isinstance(vosk_val, (int, float)) and not np.isnan(vosk_val):
                    vosk_val = f"{vosk_val:.4f}s"

            # Handle NaN and None values
            if whisper_val is None or (
                isinstance(whisper_val, float) and np.isnan(whisper_val)
            ):
                whisper_val = "N/A"
            if wav2vec2_val is None or (
                isinstance(wav2vec2_val, float) and np.isnan(wav2vec2_val)
            ):
                wav2vec2_val = "N/A"
            if vosk_val is None or (isinstance(vosk_val, float) and np.isnan(vosk_val)):
                vosk_val = "N/A"

            # Truncate long text
            if isinstance(whisper_val, str) and len(whisper_val) > 22:
                whisper_val = whisper_val[:19] + "..."
            if isinstance(wav2vec2_val, str) and len(wav2vec2_val) > 22:
                wav2vec2_val = wav2vec2_val[:19] + "..."
            if isinstance(vosk_val, str) and len(vosk_val) > 22:
                vosk_val = vosk_val[:19] + "..."

            print(f"{metric:<12} {whisper_val:<25} {wav2vec2_val:<25} {vosk_val:<25}")

        print("-" * 100)


def create_comparison_chart(results_df, output_path="model_comparison.png"):
    """Create a bar chart comparing the performance of the models."""
    try:
        import matplotlib.pyplot as plt

        # Calculate mean metrics for each model
        models = []
        wer_values = []
        mer_values = []
        wil_values = []
        time_values = []

        # Add Whisper metrics
        if "whisper_wer" in results_df.columns:
            models.append("Whisper")
            wer_values.append(results_df["whisper_wer"].mean())
            mer_values.append(results_df["whisper_mer"].mean())
            wil_values.append(results_df["whisper_wil"].mean())
            time_values.append(results_df["whisper_time"].mean())

        # Add Wav2Vec2 metrics
        if "wav2vec2_wer" in results_df.columns:
            models.append("Wav2Vec2")
            wer_values.append(results_df["wav2vec2_wer"].mean())
            mer_values.append(results_df["wav2vec2_mer"].mean())
            wil_values.append(results_df["wav2vec2_wil"].mean())
            time_values.append(results_df["wav2vec2_time"].mean())

        # Add Vosk metrics
        if "vosk_wer" in results_df.columns:
            models.append("Vosk")
            wer_values.append(results_df["vosk_wer"].mean())
            mer_values.append(results_df["vosk_mer"].mean())
            wil_values.append(results_df["vosk_wil"].mean())
            time_values.append(results_df["vosk_time"].mean())

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Error metrics plot
        x = np.arange(len(models))
        width = 0.25

        ax1.bar(x - width, wer_values, width, label="WER", color="#3274A1")
        ax1.bar(x, mer_values, width, label="MER", color="#E1812C")
        ax1.bar(x + width, wil_values, width, label="WIL", color="#3A923A")

        ax1.set_ylabel("Error Rate")
        ax1.set_title("Error Metrics Comparison (Normalized)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Processing time plot
        ax2.bar(models, time_values, color="#C03D3E")
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Average Processing Time")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Comparison chart saved to {output_path}")

    except ImportError:
        print("Matplotlib not installed. Install with:")
        print("pip install matplotlib")
        print("Skipping chart creation.")


def main():
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio")  # Directory with audio files
    reference_path = "transcript.csv"  # CSV with columns: file_id, transcript
    whisper_model_name = "base"  # Choose from: tiny, base, small, medium, large

    # Set absolute path to Vosk model
    vosk_model_path = os.path.join(current_dir, "vosk-model-en-us-0.22-lgraph")
    print(f"Using Vosk model at: {vosk_model_path}")

    output_path = "speech_recognition_comparison.csv"

    # Run comparison
    results = compare_speech_recognition_models(
        audio_dir, reference_path, whisper_model_name, vosk_model_path
    )

    # Save results
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Create and display pivot table
    print("\nDetailed Comparison Table:")
    pivot_table = create_pivot_table(results)
    print_readable_table(pivot_table)

    # Save pivot table as CSV for easy viewing in spreadsheet software
    pivot_output = "speech_recognition_pivot.csv"
    pivot_table.to_csv(pivot_output, index=False)
    print(f"\nPivot table saved to {pivot_output}")

    # Create comparison chart
    create_comparison_chart(results)

    # Print explanation of fair comparison
    print("\nNOTE: This script uses text normalization to ensure fair comparison:")
    print("- All text is converted to lowercase")
    print("- Punctuation is removed")
    print("- Special handling for contractions and possessives")
    print("- Standardized spacing")
    print(
        "This provides a more accurate representation of each model's ability to recognize speech content rather than formatting."
    )


if __name__ == "__main__":
    main()
