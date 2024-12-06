#!/bin/bash
python3 -m pip install -r requirements.txt
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
