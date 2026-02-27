import logging
from fastapi import FastAPI
from app.webhooks.whatsapp import router as whatsapp_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Voice Bot")

# Include routers
app.include_router(whatsapp_router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "WhatsApp Webhook Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
