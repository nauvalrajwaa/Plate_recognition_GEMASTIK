import base64
import cv2
import requests
import ultralytics
from openai import OpenAI
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Cek environment ultralytics
ultralytics.checks()

# Load YOLO model (ganti dengan model Anda kalau custom)
MODEL_PATH = "anpr-demo-model.pt"

# Setup OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_KEY_HERE")

# Prompt OCR
PROMPT = """
Can you extract the vehicle number plate text inside the image?
If you are not able to extract text, please respond with None.
Only output text, please.
If any text character is not from the English language, replace it with a dot (.).
"""

def extract_text(base64_encoded_data: str) -> str:
    """Panggil GPT OCR untuk ekstraksi teks plat nomor dari gambar base64."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_data}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def download_video(video_url: str, save_path: str = "input_video.mp4") -> str:
    """Download video dari URL database ke file lokal sementara."""
    r = requests.get(video_url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def process_video(video_path: str, output_path: str = "anpr-output.avi"):
    """Proses video: deteksi plat nomor + OCR + simpan hasil video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file: {video_path}"

    # Setup video writer
    w, h, fps = (int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ))
    video_writer = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps, (w, h))

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    padding = 10

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        results = model.predict(im0)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()

        ann = Annotator(im0, line_width=3)

        for cls, box in zip(clss, boxes):
            height, width, _ = im0.shape
            x1 = max(int(box[0]) - padding, 0)
            y1 = max(int(box[1]) - padding, 0)
            x2 = min(int(box[2]) + padding, width)
            y2 = min(int(box[3]) + padding, height)

            # Crop plat
            crop = im0[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Encode ke base64
            base64_im0 = base64.b64encode(cv2.imencode(".jpg", crop)[1]).decode("utf-8")

            # OCR GPT
            response_text = extract_text(base64_im0)
            print(f"Extracted text: {response_text}")

            # Tampilkan hasil
            ann.box_label(box, label=str(response_text), color=colors(int(cls), True))

        # Tulis frame
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    print(f"Video hasil disimpan di: {output_path}")

def gcp_trigger(video_url: str):
    """Fungsi utama yang dipanggil ketika ada video baru di database (URL)."""
    print(f"[INFO] Mulai proses video dari URL: {video_url}")
    local_path = download_video(video_url, "input_video.mp4")
    process_video(local_path, "anpr-output.avi")
    print("[INFO] Selesai memproses video.")

# ==== Contoh pemanggilan (simulate trigger) ====
if __name__ == "__main__":
    # Misalnya URL dari database
    test_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4"
    gcp_trigger(test_url)
