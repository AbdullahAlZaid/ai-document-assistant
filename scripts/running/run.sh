#!/bin/bash
cd "$(dirname "$0")/../.." && PYTHONPATH=. python -m streamlit run ui/app.py
