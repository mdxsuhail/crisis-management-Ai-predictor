@echo off
echo =======================================================================
echo          CRISIS MANAGEMENT LLM AI PREDICTOR (100% FREE MODE)
echo =======================================================================
echo.

echo Step 1/4: Checking Python dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements.
    pause
    exit /b %errorlevel%
)

echo.
echo Step 2/4: Cleaning and merging datasets...
python clean_and_merge.py

echo.
echo Step 3/4: Training Machine Learning model on historical dataset...
python ml_predictor.py

echo.
echo Step 4/4: Launching Streamlit Web Dashboard...
echo Dashboard opening at http://localhost:8501
streamlit run app.py

pause
