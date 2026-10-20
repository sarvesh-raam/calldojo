# Public demo image (deployed on Render).
# GROQ_API_KEY is set in the host's environment settings; it is never in the image or the repo.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Usage caps for the public link (see limits.py); override any of them in the host's settings
ENV DEMO_MODE=1 \
    DEMO_VIDEO_URL=https://drive.google.com/file/d/1-hEmO5fhhHWwHhXwAcqHx-mJpJoq1-3h/view?usp=sharing \
    PYTHONUNBUFFERED=1

# Render sets PORT; default to 7860 elsewhere. One worker: calls and limits live in memory.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}
