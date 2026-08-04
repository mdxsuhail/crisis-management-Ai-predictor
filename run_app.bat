@echo off
echo =======================================================================
echo          CRISIS MANAGEMENT LLM AI PREDICTOR (100% FREE MODE)
echo =======================================================================
echo.

echo Step 1/6: Checking Python dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements.
    pause
    exit /b %errorlevel%
)

echo.
echo Step 2/6: Cleaning and merging datasets...
python clean_and_merge.py
if %errorlevel% neq 0 (
    echo Error running clean_and_merge.py.
    pause
    exit /b %errorlevel%
)

echo.
echo Step 3/6: Calculating historical crisis impact metrics...
python analyze_crises.py
if %errorlevel% neq 0 (
    echo Error running analyze_crises.py.
    pause
    exit /b %errorlevel%
)

echo.
echo Step 4/6: Training Machine Learning models...
python ml_predictor.py
if %errorlevel% neq 0 (
    echo Error running ml_predictor.py.
    pause
    exit /b %errorlevel%
)

echo.
echo Step 5/6: Indexing crisis financial news headlines...
python crisis_news_indexer.py
if %errorlevel% neq 0 (
    echo Error running crisis_news_indexer.py.
    pause
    exit /b %errorlevel%
)

echo.
echo Step 6/6: Launching Streamlit Web Dashboard...
echo Dashboard opening at http://localhost:8501
streamlit run app.py

pause
