@echo off
echo Setting up environment...
call .venv\Scripts\activate
set KMP_DUPLICATE_LIB_OK=TRUE
echo Starting YouTube Auto-Clipper...
python youtube_auto_clipper.py %*
echo Done!
