#main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
import os

# Ensure the parent directory of 'app' is in the path
# This allows 'import app.xxx' to work regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Absolute imports are more stable for Uvicorn on Windows
from app.geospatial import find_safe_route, get_point_risk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SafeRoute Bangalore API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    hour: int
    day: str

class RiskRequest(BaseModel):
    lat: float
    lon: float
    hour: int
    day: str

@app.get("/")
async def root():
    return {"message": "SafeRoute Bangalore API is running"}

@app.post("/get-safe-route")
async def get_route(req: RouteRequest):
    logger.info(f"--- ROUTE REQUEST: ({req.start_lat}, {req.start_lon}) to ({req.end_lat}, {req.end_lon}) ---")
    try:
        result = find_safe_route(
            orig_lat=req.start_lat,
            orig_lon=req.start_lon,
            dest_lat=req.end_lat,
            dest_lon=req.end_lon,
            hour=req.hour,
            day=req.day
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        logger.error(f"Routing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-point-risk")
async def get_risk(req: RiskRequest):
    try:
        return get_point_risk(req.lat, req.lon, req.hour, req.day)
    except Exception as e:
        logger.error(f"Risk inspection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Calculation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)