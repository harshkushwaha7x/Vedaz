# Hosting the Fine-Tuned Vedaz Model on a VPS with vLLM

This guide covers the complete process: picking the right VPS, installing vLLM,
uploading the merged model, starting the server, and running it reliably in production.

---

## 1. VPS Requirements

### Minimum hardware (Qwen2.5-7B merged, bfloat16)

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | 24 GB+ |
| GPU | RTX 4090, A10G | A100 40GB, H100 |
| System RAM | 32 GB | 64 GB |
| Disk | 60 GB SSD | 150 GB NVMe |
| CPU | 8 cores | 16 cores |
| Network | 1 Gbps | 10 Gbps |

> **Smaller models:** Qwen2.5-3B fits on a 10GB GPU (RTX 3080/4070).
> Qwen3-4B fits on 12GB. These are good starting points for a cost-conscious deployment.

### Recommended GPU cloud providers

| Provider | GPU options | Notes |
|---|---|---|
| Lambda Labs | A10 (24GB), A100 | Hourly billing, straightforward setup |
| Vast.ai | Wide range including 3090/4090 | Cheapest option, community servers |
| RunPod | A10G, A100, H100 | Good UI, persistent storage |
| Google Cloud | A100, T4 | Higher cost, enterprise SLAs |
| AWS EC2 | g5 (A10G), p3/p4 | Enterprise, complex setup |

For a production Vedaz deployment, **Lambda Labs A10 (24GB)** or **RunPod A100** is
the practical sweet spot between cost and reliability.

---

## 2. Initial VPS Setup

SSH into your VPS and run the following. These commands are written for **Ubuntu 22.04**
(the most common GPU cloud OS).

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y git wget curl build-essential python3-pip python3-venv \
    nvidia-cuda-toolkit nvtop htop screen

# Verify GPU is visible to the OS
nvidia-smi
# Expected output: shows your GPU model, driver version, CUDA version
```

---

## 3. Install Python Environment

Always use a virtual environment — it keeps dependencies isolated and makes
rollbacks clean.

```bash
# Create a project directory
mkdir -p ~/vedaz-server && cd ~/vedaz-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 — match your driver version; run `nvidia-smi` to check)
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
# vLLM has its own CUDA kernels; the right version matters.
pip install vllm==0.6.3

# Verify vLLM installed correctly
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

> **Note on versions:** vLLM releases frequently. If `vllm==0.6.3` is unavailable,
> use `pip install vllm` for the latest, then check the Qwen model is supported
> at https://docs.vllm.ai/en/latest/models/supported_models.html

---

## 4. Upload the Merged Model to the VPS

The merged model produced by `merge_model.py` is a folder of `.safetensors` files
(roughly 14GB for Qwen2.5-7B). Three upload options:

### Option A — rsync from your training machine (simplest)

```bash
# Run this from your LOCAL machine (the one where you trained the model)
rsync -avzP --progress \
    output/vedaz-qwen-merged/ \
    user@your-vps-ip:~/vedaz-server/models/vedaz-qwen-merged/
```

### Option B — Push to Hugging Face Hub, pull on VPS (cleanest for teams)

```bash
# On training machine: push to HF Hub (requires HF account + huggingface-cli login)
pip install huggingface_hub
huggingface-cli login          # paste your HF token
huggingface-cli upload your-username/vedaz-qwen-merged output/vedaz-qwen-merged/

# On VPS: pull from HF Hub
pip install huggingface_hub
huggingface-cli download your-username/vedaz-qwen-merged \
    --local-dir ~/vedaz-server/models/vedaz-qwen-merged/
```

> Set the repo to **private** on HF Hub so the model isn't publicly accessible.

### Option C — Direct download from cloud storage (fast for large models)

```bash
# Upload to S3, then on VPS:
aws s3 cp s3://your-bucket/vedaz-qwen-merged/ ~/vedaz-server/models/vedaz-qwen-merged/ \
    --recursive --no-sign-request
```

---

## 5. Start the vLLM Server

vLLM runs an OpenAI-compatible REST API server. Once started, you can call it
with the same interface as the OpenAI API — any client that supports OpenAI
also works with vLLM.

### Basic start (for testing)

```bash
cd ~/vedaz-server
source venv/bin/activate

vllm serve models/vedaz-qwen-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name vedaz-astrologer \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90
```

**What each flag does:**

| Flag | Value | Why |
|---|---|---|
| `--host 0.0.0.0` | bind all interfaces | Accept connections from outside, not just localhost |
| `--port 8000` | 8000 | Standard port; change if occupied |
| `--served-model-name` | vedaz-astrologer | The model name clients use in API calls |
| `--max-model-len` | 4096 | Max total tokens per request (prompt + generation) |
| `--dtype bfloat16` | bfloat16 | Match the precision the model was saved in |
| `--gpu-memory-utilization` | 0.90 | Use 90% of VRAM for the KV cache (leave 10% headroom) |

You should see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Test the server

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Send a chat request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astrologer",
    "messages": [
      {
        "role": "system",
        "content": "You are Vedaz'\''s AI Vedic astrologer."
      },
      {
        "role": "user",
        "content": "Meri shaadi kab hogi? DOB 15 Aug 1995, 10:30 AM, Mumbai."
      }
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

---

## 6. Run as a systemd Service (Production)

Running in a terminal works for testing but dies when you close SSH. Use
**systemd** to run vLLM as a background service that starts automatically
on reboot and restarts on crashes.

### Create the service file

```bash
sudo nano /etc/systemd/system/vedaz-vllm.service
```

Paste this (replace `YOUR_USERNAME` with your actual Linux username):

```ini
[Unit]
Description=Vedaz vLLM Inference Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/vedaz-server
Environment=PATH=/home/YOUR_USERNAME/vedaz-server/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/YOUR_USERNAME/vedaz-server/venv/bin/vllm serve \
    models/vedaz-qwen-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name vedaz-astrologer \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vedaz-vllm

[Install]
WantedBy=multi-user.target
```

### Enable and start the service

```bash
# Reload systemd to pick up the new file
sudo systemctl daemon-reload

# Enable so it starts on boot
sudo systemctl enable vedaz-vllm

# Start now
sudo systemctl start vedaz-vllm

# Check status
sudo systemctl status vedaz-vllm

# Watch live logs
sudo journalctl -u vedaz-vllm -f
```

---

## 7. (Optional) Nginx Reverse Proxy + HTTPS

Exposing port 8000 directly works but is not secure for production. Nginx
sits in front of vLLM, handles HTTPS termination, and lets you add API
key authentication at the nginx layer.

```bash
sudo apt install -y nginx certbot python3-certbot-nginx

# Point your domain (e.g. api.vedaz.in) to your VPS IP in DNS first, then:
sudo certbot --nginx -d api.vedaz.in
```

Create `/etc/nginx/sites-available/vedaz-api`:

```nginx
server {
    listen 443 ssl;
    server_name api.vedaz.in;

    ssl_certificate     /etc/letsencrypt/live/api.vedaz.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.vedaz.in/privkey.pem;

    # Simple API key check — replace with your actual key
    # For proper auth, use a dedicated API gateway instead
    if ($http_authorization != "Bearer YOUR_SECRET_KEY") {
        return 401;
    }

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 300s;     # LLM responses can take time
        proxy_send_timeout 300s;
    }
}

server {
    listen 80;
    server_name api.vedaz.in;
    return 301 https://$server_name$request_uri;
}
```

```bash
sudo ln -s /etc/nginx/sites-available/vedaz-api /etc/nginx/sites-enabled/
sudo nginx -t          # test config
sudo systemctl reload nginx
```

---

## 8. Calling the Server from Your Application

The vLLM server is OpenAI-compatible, so you can call it with the official
OpenAI Python client by just pointing `base_url` at your VPS:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.vedaz.in/v1",   # your VPS URL
    api_key="YOUR_SECRET_KEY",            # the Bearer key from nginx
)

response = client.chat.completions.create(
    model="vedaz-astrologer",
    messages=[
        {
            "role": "system",
            "content": "You are Vedaz's AI Vedic astrologer..."
        },
        {
            "role": "user",
            "content": "Meri shaadi kab hogi?"
        }
    ],
    max_tokens=600,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

---

## 9. Useful vLLM Flags for Production

```bash
# Multi-GPU: spread across 2 GPUs (tensor parallelism)
vllm serve models/vedaz-qwen-merged \
    --tensor-parallel-size 2 \
    ...

# Streaming responses (returns tokens as they're generated)
# No extra flag needed — vLLM supports SSE streaming out of the box.
# Use stream=True in the Python client.

# Limit concurrent requests to avoid OOM
vllm serve models/vedaz-qwen-merged \
    --max-num-seqs 16 \    # max simultaneous requests
    ...

# Quantize at serving time (saves VRAM, small quality drop)
vllm serve models/vedaz-qwen-merged \
    --quantization awq \   # requires model already saved in AWQ format
    ...
```

---

## 10. Monitoring

```bash
# GPU utilization and VRAM usage (live)
watch -n 1 nvidia-smi

# vLLM logs
sudo journalctl -u vedaz-vllm -f

# Requests per second (vLLM exposes Prometheus metrics)
curl http://localhost:8000/metrics | grep vllm_request
```

---

## Full Checklist

- [ ] GPU VPS provisioned (≥16GB VRAM for 7B model)
- [ ] Ubuntu 22.04, CUDA 12.x, nvidia-smi shows GPU
- [ ] Python venv created, PyTorch + vLLM installed
- [ ] Merged model uploaded to `~/vedaz-server/models/vedaz-qwen-merged/`
- [ ] vLLM starts and health check returns 200
- [ ] Test chat request returns a sensible Vedaz-voice response
- [ ] systemd service created, enabled, running
- [ ] (Optional) Nginx + HTTPS configured
- [ ] Monitoring set up (nvidia-smi, journalctl)
