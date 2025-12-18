
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import uploads, experiments
from app.api.control import router as control_router

app = FastAPI(title="Battery RL Backend")

# 🔥 CORS FIX (THIS IS THE ROOT CAUSE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5178",  # Angular frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(uploads.router)
app.include_router(experiments.router)
app.include_router(control_router)

@app.get("/")
def root():
    return {"message": "Battery RL API running"}
