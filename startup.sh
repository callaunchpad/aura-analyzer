#!/bin/bash
python3 -m pip install -r requirements.txt
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app
