FROM python:3.11-slim

WORKDIR /srv
# opencv-python (a simple-lama-inpainting dependency) links libGL and glib.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py mcp_server.py fit_alpha.py fit_detector.py test_mask.py logo_mask_1280.png logo_k.png logo_b.png logo_detector.png logo_detector.json ./
RUN python test_mask.py
# Pull the LaMa weights at build time so the first request does not download,
# and pin them: the file comes from a GitHub release without a checksum.
RUN python -c "from simple_lama_inpainting.utils import download_model; from simple_lama_inpainting.models.model import LAMA_MODEL_URL; print(download_model(LAMA_MODEL_URL))" \
 && echo "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c  /root/.cache/torch/hub/checkpoints/big-lama.pt" | sha256sum -c -

# Torch with one thread per core thrashes on small windows: 14 threads took
# 11 s per photo in a VM, 4 threads 1-3 s. Override for a big dedicated box.
ENV HOST=0.0.0.0 PORT=8765 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
EXPOSE 8765
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"
CMD ["python", "app.py"]
