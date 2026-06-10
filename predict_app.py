from __future__ import annotations

import argparse
import base64
import html
import io
import mimetypes
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlparse

warnings.filterwarnings(
    "ignore",
    message="'cgi' is deprecated.*",
    category=DeprecationWarning,
)
import cgi

import numpy as np
import torch
from PIL import Image, ImageOps

from class_presets import selected_classes_for_preset
from model.mlp import MLPMixerClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DOG_INFO = {
    "Afghan Hound": {
        "summary": "Chó săn dáng cao, lông dài mượt, nổi bật bởi vẻ thanh lịch và tính độc lập.",
        "traits": "Độc lập, nhanh nhẹn, cần chải lông thường xuyên.",
        "wikipedia": "https://en.wikipedia.org/wiki/Afghan_Hound",
        "akc": "https://www.akc.org/dog-breeds/afghan-hound/",
    },
    "Basset Hound": {
        "summary": "Chó săn chân ngắn, tai dài, khứu giác tốt và tính cách hiền.",
        "traits": "Điềm tĩnh, thân thiện, dễ tăng cân nếu ít vận động.",
        "wikipedia": "https://en.wikipedia.org/wiki/Basset_Hound",
        "akc": "https://www.akc.org/dog-breeds/basset-hound/",
    },
    "Bull Terrier": {
        "summary": "Giống chó cơ bắp, đầu hình quả trứng, năng lượng cao.",
        "traits": "Tinh nghịch, mạnh mẽ, cần vận động và huấn luyện đều.",
        "wikipedia": "https://en.wikipedia.org/wiki/Bull_Terrier",
        "akc": "https://www.akc.org/dog-breeds/bull-terrier/",
    },
    "Chihuahua": {
        "summary": "Một trong những giống chó nhỏ nhất, lanh lợi và rất gắn bó với chủ.",
        "traits": "Nhỏ, cảnh giác, hợp không gian sống nhỏ.",
        "wikipedia": "https://en.wikipedia.org/wiki/Chihuahua_(dog)",
        "akc": "https://www.akc.org/dog-breeds/chihuahua/",
    },
    "Chow Chow": {
        "summary": "Chó lông dày, dáng chắc, nổi tiếng với bờm quanh cổ và lưỡi xanh đen.",
        "traits": "Điềm đạm, độc lập, cần chăm sóc lông kỹ.",
        "wikipedia": "https://en.wikipedia.org/wiki/Chow_Chow",
        "akc": "https://www.akc.org/dog-breeds/chow-chow/",
    },
    "Dalmatian": {
        "summary": "Giống chó thân hình thể thao với bộ lông trắng đốm đen/nâu rất đặc trưng.",
        "traits": "Năng động, bền bỉ, cần vận động nhiều.",
        "wikipedia": "https://en.wikipedia.org/wiki/Dalmatian_dog",
        "akc": "https://www.akc.org/dog-breeds/dalmatian/",
    },
    "Great Dane": {
        "summary": "Giống chó khổng lồ, cao lớn nhưng thường hiền và thân thiện.",
        "traits": "Rất lớn, điềm tĩnh, cần không gian và kiểm soát vận động.",
        "wikipedia": "https://en.wikipedia.org/wiki/Great_Dane",
        "akc": "https://www.akc.org/dog-breeds/great-dane/",
    },
    "Greyhound": {
        "summary": "Chó săn tốc độ cao, thân thon, chân dài, nổi bật về khả năng chạy.",
        "traits": "Nhanh, nhẹ nhàng, thích chạy ngắn nhưng cũng ngủ nhiều.",
        "wikipedia": "https://en.wikipedia.org/wiki/Greyhound",
        "akc": "https://www.akc.org/dog-breeds/greyhound/",
    },
    "Pembroke Welsh Corgi": {
        "summary": "Chó chăn gia súc chân ngắn, thân dài, thông minh và hoạt bát.",
        "traits": "Thông minh, lanh lợi, dễ huấn luyện.",
        "wikipedia": "https://en.wikipedia.org/wiki/Pembroke_Welsh_Corgi",
        "akc": "https://www.akc.org/dog-breeds/pembroke-welsh-corgi/",
    },
    "Poodle": {
        "summary": "Giống chó thông minh, lông xoăn, có nhiều kích cỡ khác nhau.",
        "traits": "Thông minh, dễ huấn luyện, cần chăm sóc lông định kỳ.",
        "wikipedia": "https://en.wikipedia.org/wiki/Poodle",
        "akc": "https://www.akc.org/dog-breeds/poodle-standard/",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple image prediction app for MLP-Mixer.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint/mlp_mixer_diverse10_best.pt"),
    )
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--sample-data-dir", type=Path, default=Path("data/merged_dog_dataset_v2"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def find_sample_images(data_dir: Path, labels: list[str]) -> dict[str, Path]:
    samples = {}
    for label in labels:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples[label] = path
                break
    return samples


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return requested


def load_labels(checkpoint: dict, label_file: Path | None) -> list[str]:
    if label_file is not None:
        return read_label_file(label_file)

    args = checkpoint.get("args", {})
    artifacts_dir = Path(args.get("artifacts_dir", "artifacts"))
    selected_classes_path = artifacts_dir / "selected_classes.txt"
    if selected_classes_path.exists():
        return read_label_file(selected_classes_path)

    preset = args.get("class_preset", "all")
    preset_labels = selected_classes_for_preset(preset)
    if preset_labels is not None:
        return list(preset_labels)

    raise ValueError(
        "Cannot infer labels. Pass --label-file artifacts/selected_classes.txt."
    )


def read_label_file(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"Label file is empty: {path}")
    return labels


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    args = dict(checkpoint.get("args", {}))
    return SimpleNamespace(**args)


def build_model(checkpoint: dict, device: str) -> MLPMixerClassifier:
    args = checkpoint_args(checkpoint)
    input_size = int(checkpoint["input_size"])
    num_classes = int(checkpoint["num_classes"])

    model = MLPMixerClassifier(
        input_size=input_size,
        num_classes=num_classes,
        image_size=getattr(args, "image_size", 64),
        channels=getattr(args, "channels", 3),
        patch_size=getattr(args, "patch_size", 4),
        hidden_size=getattr(args, "hidden_size", 256),
        depth=getattr(args, "depth", 4),
        expansion=getattr(args, "expansion", 2),
        token_mlp_size=getattr(args, "token_mlp_size", 128),
        channel_mlp_size=getattr(args, "channel_mlp_size", 0),
        dropout=getattr(args, "dropout", 0.15),
        epochs=1,
        batch_size=getattr(args, "batch_size", 128),
        learning_rate=getattr(args, "learning_rate", 3e-4),
        weight_decay=getattr(args, "weight_decay", 0.0),
        optimizer_name=getattr(args, "optimizer", "adamw"),
        label_smoothing=getattr(args, "label_smoothing", 0.0),
        feature_noise=0.0,
        feature_drop=0.0,
        crop_padding=0,
        hflip_prob=0.0,
        erase_prob=0.0,
        erase_scale=0.0,
        mixup_alpha=0.0,
        ema_decay=0.0,
        warmup_epochs=0,
        grad_clip=0.0,
        drop_path=getattr(args, "drop_path", 0.0),
        layer_scale=getattr(args, "layer_scale", 0.1),
        patience=1,
        device=device,
    )
    model.network.load_state_dict(checkpoint["model_state"])
    model.network.to(device)
    model.network.eval()
    return model


def infer_image_size(input_size: int, channels: int, requested_size: int) -> int:
    if requested_size > 0:
        return requested_size
    inferred_size = int(round((input_size / channels) ** 0.5))
    if inferred_size * inferred_size * channels != input_size:
        raise ValueError("Cannot infer image size from checkpoint.")
    return inferred_size


def preprocess_image(image_bytes: bytes, image_size: int, channels: int) -> torch.Tensor:
    color_mode = "L" if channels == 1 else "RGB"
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert(color_mode)
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        features = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    return torch.from_numpy(features).float().unsqueeze(0)


@torch.inference_mode()
def predict(
    model: MLPMixerClassifier,
    image_bytes: bytes,
    labels: list[str],
    device: str,
    image_size: int,
    channels: int,
    top_k: int,
) -> list[tuple[str, float]]:
    features = preprocess_image(image_bytes, image_size, channels).to(device)
    logits = model.network(features)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    k = min(top_k, len(labels), probabilities.numel())
    scores, indices = probabilities.topk(k)
    return [(labels[index.item()], 100.0 * score.item()) for score, index in zip(scores, indices)]


def image_data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_breed_info_cards(sample_images: dict[str, Path]) -> str:
    cards = []
    for label, info in DOG_INFO.items():
        image_html = (
            f"<img class='result-image' src='/sample?label={quote(label)}' "
            f"alt='{html.escape(label)}'>"
            if label in sample_images
            else "<div class='result-image placeholder'>No image</div>"
        )
        links = (
            f"<a href='{html.escape(info['wikipedia'])}' target='_blank' "
            "rel='noopener noreferrer'>Wikipedia</a>"
            f"<a href='{html.escape(info['akc'])}' target='_blank' "
            "rel='noopener noreferrer'>AKC</a>"
        )
        cards.append(
            f"""
            <article class="result-card">
              {image_html}
              <div class="result-main">
                <div class="result-topline">
                  <h3>{html.escape(label)}</h3>
                </div>
                <p>{html.escape(info['summary'])}</p>
                <p class="traits">{html.escape(info['traits'])}</p>
                <div class="links">{links}</div>
              </div>
            </article>
            """
        )
    return "\n".join(cards)


def render_page(
    results: list[tuple[str, float]] | None,
    error: str | None,
    sample_images: dict[str, Path],
    uploaded_preview: str | None,
) -> bytes:
    result_html = ""
    if error:
        result_html = f"<p class='error'>{html.escape(error)}</p>"
    elif results:
        cards = []
        for rank, (label, score) in enumerate(results, start=1):
            info = DOG_INFO.get(label, {})
            summary = info.get("summary", "Chưa có mô tả ngắn cho giống chó này.")
            traits = info.get("traits", "")
            wiki = info.get("wikipedia", "")
            akc = info.get("akc", "")
            image_html = (
                f"<img class='result-image' src='/sample?label={quote(label)}' "
                f"alt='{html.escape(label)}'>"
                if label in sample_images
                else "<div class='result-image placeholder'>No image</div>"
            )
            links = ""
            if wiki:
                links += (
                    f"<a href='{html.escape(wiki)}' target='_blank' "
                    "rel='noopener noreferrer'>Wikipedia</a>"
                )
            if akc:
                links += (
                    f"<a href='{html.escape(akc)}' target='_blank' "
                    "rel='noopener noreferrer'>AKC</a>"
                )
            cards.append(
                f"""
                <article class="result-card">
                  <div class="rank">#{rank}</div>
                  {image_html}
                  <div class="result-main">
                    <div class="result-topline">
                      <h3>{html.escape(label)}</h3>
                      <strong>{score:.2f}%</strong>
                    </div>
                    <div class="bar"><span style="width: {score:.2f}%"></span></div>
                    <p>{html.escape(summary)}</p>
                    <p class="traits">{html.escape(traits)}</p>
                    <div class="links">{links}</div>
                  </div>
                </article>
                """
            )
        cards_html = "\n".join(cards)
        result_html = f"""
        <section class="results">
          <div class="section-title">
            <span>Kết quả dự đoán</span>
            <small>Top {len(results)} nhãn có xác suất cao nhất</small>
          </div>
          {cards_html}
        </section>
        """

    preview_html = ""
    if uploaded_preview:
        preview_html = f"<img id='preview' class='preview' src='{uploaded_preview}' alt='Ảnh đã chọn'>"

    page = f"""
    <!doctype html>
    <html lang="vi">
    <head>
      <meta charset="utf-8">
      <title>Dog Breed MLP-Mixer</title>
      <style>
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: Inter, Arial, sans-serif;
          color: #172033;
          background:
            radial-gradient(circle at 15% 15%, #dbeafe 0, transparent 28%),
            radial-gradient(circle at 90% 0%, #fef3c7 0, transparent 24%),
            linear-gradient(135deg, #f8fafc, #eef2ff);
        }}
        .page {{ max-width: 1120px; margin: 0 auto; padding: 40px 20px 56px; }}
        .hero {{
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
          gap: 24px;
          align-items: stretch;
        }}
        .panel {{
          background: rgba(255, 255, 255, 0.86);
          border: 1px solid rgba(148, 163, 184, 0.28);
          border-radius: 24px;
          box-shadow: 0 20px 60px rgba(15, 23, 42, 0.10);
          backdrop-filter: blur(10px);
        }}
        .intro {{ padding: 32px; }}
        .badge {{
          display: inline-flex;
          padding: 7px 12px;
          border-radius: 999px;
          color: #1d4ed8;
          background: #dbeafe;
          font-size: 13px;
          font-weight: 700;
        }}
        h1 {{ margin: 18px 0 12px; font-size: 42px; line-height: 1.04; }}
        .subtitle {{ color: #52627a; font-size: 17px; line-height: 1.6; }}
        .stats {{
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin-top: 24px;
        }}
        .stat {{
          padding: 14px;
          border-radius: 16px;
          background: #f8fafc;
          border: 1px solid #e2e8f0;
        }}
        .stat strong {{ display: block; font-size: 22px; }}
        .stat span {{ color: #64748b; font-size: 13px; }}
        form {{ padding: 22px; }}
        .drop-zone {{
          min-height: 315px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          border: 2px dashed #93c5fd;
          border-radius: 20px;
          padding: 26px;
          text-align: center;
          color: #475569;
          cursor: pointer;
          background: linear-gradient(180deg, #eff6ff, #ffffff);
          transition: 0.18s ease;
        }}
        .drop-zone.dragover {{
          background: #dbeafe;
          border-color: #2563eb;
          transform: translateY(-2px);
        }}
        .upload-icon {{
          width: 58px;
          height: 58px;
          display: grid;
          place-items: center;
          border-radius: 18px;
          background: #2563eb;
          color: white;
          font-size: 28px;
          margin-bottom: 16px;
        }}
        .hint {{ font-size: 14px; color: #64748b; }}
        button {{
          width: 100%;
          margin-top: 14px;
          padding: 13px 16px;
          border: 0;
          border-radius: 14px;
          background: linear-gradient(135deg, #2563eb, #7c3aed);
          color: white;
          font-weight: 800;
          cursor: pointer;
          box-shadow: 0 12px 24px rgba(37, 99, 235, 0.26);
        }}
        button:hover {{ filter: brightness(1.03); }}
        .preview {{
          max-width: 100%;
          max-height: 230px;
          margin-top: 16px;
          border-radius: 16px;
          box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        }}
        .results {{
          margin-top: 28px;
          padding: 24px;
          border-radius: 24px;
          background: rgba(255, 255, 255, 0.88);
          border: 1px solid rgba(148, 163, 184, 0.28);
          box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
        }}
        .section-title {{
          display: flex;
          justify-content: space-between;
          align-items: end;
          gap: 16px;
          margin-bottom: 16px;
        }}
        .section-title span {{ font-size: 24px; font-weight: 900; }}
        .section-title small {{ color: #64748b; }}
        .result-card {{
          position: relative;
          display: grid;
          grid-template-columns: 116px minmax(0, 1fr);
          gap: 18px;
          padding: 16px;
          margin-top: 14px;
          border-radius: 20px;
          background: #ffffff;
          border: 1px solid #e2e8f0;
        }}
        .rank {{
          position: absolute;
          top: 12px;
          left: 12px;
          z-index: 1;
          padding: 5px 8px;
          border-radius: 999px;
          background: rgba(15, 23, 42, 0.76);
          color: white;
          font-size: 12px;
          font-weight: 800;
        }}
        .result-image {{
          width: 116px;
          height: 116px;
          object-fit: cover;
          border-radius: 16px;
          background: #e2e8f0;
        }}
        .placeholder {{
          display: grid;
          place-items: center;
          color: #64748b;
          font-size: 13px;
        }}
        .result-topline {{
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: center;
        }}
        .result-topline h3 {{ margin: 0; font-size: 21px; }}
        .result-topline strong {{ color: #2563eb; font-size: 22px; }}
        .bar {{
          height: 10px;
          margin: 10px 0 12px;
          overflow: hidden;
          border-radius: 999px;
          background: #e2e8f0;
        }}
        .bar span {{
          display: block;
          height: 100%;
          border-radius: inherit;
          background: linear-gradient(90deg, #2563eb, #7c3aed);
        }}
        .result-main p {{ margin: 8px 0; color: #475569; line-height: 1.45; }}
        .traits {{ font-weight: 700; color: #334155 !important; }}
        .links {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
        .links a {{
          text-decoration: none;
          padding: 8px 10px;
          border-radius: 999px;
          color: #1d4ed8;
          background: #dbeafe;
          font-size: 13px;
          font-weight: 800;
        }}
        .error {{ color: #b00020; }}
        #imageInput {{ display: none; }}
        @media (max-width: 820px) {{
          .hero {{ grid-template-columns: 1fr; }}
          h1 {{ font-size: 34px; }}
          .stats {{ grid-template-columns: 1fr; }}
          .result-card {{ grid-template-columns: 92px 1fr; }}
          .result-image {{ width: 92px; height: 92px; }}
          .section-title {{ display: block; }}
        }}
      </style>
    </head>
    <body>
      <main class="page">
        <section class="hero">
          <div class="panel intro">
            <span class="badge">MLP-Mixer Dog Classifier</span>
            <h1>Dự đoán giống chó từ ảnh</h1>
            <p class="subtitle">
              Tải ảnh, kéo-thả hoặc paste trực tiếp để mô hình MLP-Mixer dự đoán giống chó.
              Kết quả hiển thị xác suất, ảnh mẫu từ dataset và link thông tin chi tiết.
            </p>
            <div class="stats">
              <div class="stat"><strong>10</strong><span>giống chó</span></div>
              <div class="stat"><strong>Top-K</strong><span>xếp hạng xác suất</span></div>
              <div class="stat"><strong>64x64</strong><span>preprocess ảnh</span></div>
            </div>
          </div>
          <form id="uploadForm" class="panel" method="post" enctype="multipart/form-data">
            <div id="dropZone" class="drop-zone">
              <div class="upload-icon">↑</div>
              <strong>Kéo ảnh vào đây, click để chọn ảnh, hoặc Ctrl+V để paste ảnh</strong>
              <p class="hint">Hỗ trợ jpg, png, webp...</p>
              <input id="imageInput" type="file" name="image" accept="image/*" required>
              {preview_html}
            </div>
            <button type="submit">Dự đoán ngay</button>
          </form>
        </section>
        {result_html}
        <div class="results">
          <div class="section-title">
            <span>Thông tin các giống chó</span>
            <small>Link ngoài để đọc đầy đủ hơn</small>
          </div>
          {render_breed_info_cards(sample_images)}
        </div>
      </main>
      <script>
        const dropZone = document.getElementById('dropZone');
        const imageInput = document.getElementById('imageInput');

        function setFile(file) {{
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          imageInput.files = dataTransfer.files;

          const oldPreview = document.getElementById('preview');
          if (oldPreview) oldPreview.remove();
          const preview = document.createElement('img');
          preview.id = 'preview';
          preview.className = 'preview';
          preview.src = URL.createObjectURL(file);
          dropZone.appendChild(preview);
        }}

        dropZone.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', () => {{
          if (imageInput.files.length > 0) setFile(imageInput.files[0]);
        }});

        dropZone.addEventListener('dragover', (event) => {{
          event.preventDefault();
          dropZone.classList.add('dragover');
        }});
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (event) => {{
          event.preventDefault();
          dropZone.classList.remove('dragover');
          if (event.dataTransfer.files.length > 0) setFile(event.dataTransfer.files[0]);
        }});

        document.addEventListener('paste', (event) => {{
          for (const item of event.clipboardData.items) {{
            if (item.type.startsWith('image/')) {{
              const file = item.getAsFile();
              if (file) setFile(file);
              break;
            }}
          }}
        }});
      </script>
    </body>
    </html>
    """
    return page.encode("utf-8")


def make_handler(
    model: MLPMixerClassifier,
    labels: list[str],
    device: str,
    image_size: int,
    channels: int,
    top_k: int,
    sample_images: dict[str, Path],
) -> type[BaseHTTPRequestHandler]:
    class PredictHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/sample":
                self.send_sample(parsed.query)
                return
            self.send_page()

        def do_POST(self) -> None:
            results = None
            error = None
            try:
                content_type = self.headers.get("Content-Type", "")
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    },
                )
                image_field = form["image"] if "image" in form else None
                if image_field is None or not getattr(image_field, "file", None):
                    raise ValueError("Chưa chọn ảnh.")
                image_bytes = image_field.file.read()
                uploaded_preview = image_data_uri(image_bytes)
                results = predict(
                    model,
                    image_bytes,
                    labels,
                    device,
                    image_size,
                    channels,
                    top_k,
                )
            except Exception as exc:
                error = str(exc)
                uploaded_preview = None
            self.send_page(results=results, error=error, uploaded_preview=uploaded_preview)

        def send_sample(self, query: str) -> None:
            label = parse_qs(query).get("label", [""])[0]
            sample_path = sample_images.get(label)
            if sample_path is None:
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(sample_path.name)[0] or "image/jpeg"
            image_bytes = sample_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(image_bytes)))
            self.end_headers()
            self.wfile.write(image_bytes)

        def send_page(
            self,
            results: list[tuple[str, float]] | None = None,
            error: str | None = None,
            uploaded_preview: str | None = None,
        ) -> None:
            body = render_page(results, error, sample_images, uploaded_preview)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PredictHandler


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_namespace = checkpoint_args(checkpoint)
    labels = load_labels(checkpoint, args.label_file)
    if len(labels) != int(checkpoint["num_classes"]):
        raise SystemExit(
            f"Label count ({len(labels)}) != num_classes ({checkpoint['num_classes']})."
        )

    channels = getattr(checkpoint_namespace, "channels", 3)
    image_size = infer_image_size(
        int(checkpoint["input_size"]),
        channels,
        getattr(checkpoint_namespace, "image_size", 64),
    )
    model = build_model(checkpoint, device)
    sample_images = find_sample_images(args.sample_data_dir, labels)

    handler = make_handler(
        model=model,
        labels=labels,
        device=device,
        image_size=image_size,
        channels=channels,
        top_k=args.top_k,
        sample_images=sample_images,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Labels: {', '.join(labels)}")
    print(f"Sample images: {len(sample_images)}/{len(labels)} from {args.sample_data_dir}")
    print(f"Open: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
