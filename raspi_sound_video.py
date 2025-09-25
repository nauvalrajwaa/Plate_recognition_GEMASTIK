import cv2
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import os
import wave
import subprocess
from google.cloud import storage

# --- Konfigurasi ---
THRESHOLD_DB = -20        # ambang batas dB
RECORD_DURATION = 15      # detik
OUTPUT_DIR = f"/home/{os.getenv('USER')}/detections/"
BUCKET_NAME = "knalpot-brong-data"  # ganti dengan nama bucket GCP Anda

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

audio_q = queue.Queue()

# --- Audio Callback ---
def audio_callback(indata, frames, time_info, status):
    volume = np.sqrt(np.mean(indata**2))
    db = 20 * np.log10(volume + 1e-6)
    audio_q.put(db)

def start_audio_stream(samplerate=44100, channels=1, blocksize=1024):
    with sd.InputStream(channels=channels, samplerate=samplerate,
                        blocksize=blocksize, callback=audio_callback):
        while True:
            time.sleep(0.1)

# --- Rekam Audio ---
def record_audio(filename, duration=RECORD_DURATION, samplerate=44100, channels=1):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=channels, dtype="int16")
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    print(f"[INFO] Rekaman audio selesai: {filename}")

# --- Rekam Video ---
def record_video(filename, duration=RECORD_DURATION):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Rekaman video selesai: {filename}")
    return True

# --- Gabungkan Video + Audio ---
def merge_av(video_file, audio_file, output_file):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-i", audio_file,
        "-c:v", "copy", "-c:a", "aac",
        "-shortest", output_file
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[INFO] Video+Audio digabung: {output_file}")

# --- Upload ke GCS ---
def upload_to_gcs(local_file, bucket_name=BUCKET_NAME):
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"videos/{time.strftime('%Y%m%d')}/{os.path.basename(local_file)}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file)
        print(f"[INFO] File {local_file} berhasil diupload ke GCS: {blob_name}")
    except Exception as e:
        print(f"[ERROR] Upload gagal: {e}")

# --- Main Loop ---
if __name__ == "__main__":
    t = threading.Thread(target=start_audio_stream, daemon=True)
    t.start()

    print("[INFO] Sistem monitoring aktif... Tekan Ctrl+C untuk stop.")

    try:
        while True:
            db = None
            while not audio_q.empty():
                db = audio_q.get()

            if db is not None:
                print(f"Level suara: {db:.2f} dB")
                if db > THRESHOLD_DB:
                    print("[ALERT] Suara brong terdeteksi! Mulai rekaman...")

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    raw_video = os.path.join(OUTPUT_DIR, f"brong_{timestamp}_raw.mp4")
                    raw_audio = os.path.join(OUTPUT_DIR, f"brong_{timestamp}.wav")
                    final_output = os.path.join(OUTPUT_DIR, f"brong_{timestamp}.mp4")

                    # Rekam video & audio (jalan paralel lebih bagus, tapi ini sequential untuk sederhana)
                    record_video(raw_video)
                    record_audio(raw_audio)

                    # Merge jadi 1 MP4
                    merge_av(raw_video, raw_audio, final_output)

                    # Upload hasil final
                    upload_to_gcs(final_output)

                    time.sleep(5)  # cooldown

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Sistem dihentikan manual.")
