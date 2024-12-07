#!/bin/bash
python3 -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0
