import uvicorn
from backend.config import BACKEND_HOST, BACKEND_PORT

if __name__ == "__main__":
    print(f"Starting EnterpriseDoc AI Backend on http://{BACKEND_HOST}:{BACKEND_PORT}")
    uvicorn.run("backend.app:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)

