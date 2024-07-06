from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Go to /streamlit to access the Streamlit app"}

@app.get("/streamlit")
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])
    return {"message": "Streamlit app should be running on port 8501"}
