import os
from fastapi import FastAPI, Request, HTTPException, Response

app = FastAPI()

# Identify token for WhatsApp configuration
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_verify_token_123")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "WhatsApp Webhook Server is running"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    WhatsApp webhook verification endpoint.
    Meta docs: https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            # Must return the challenge as plain text
            return Response(content=challenge, media_type="text/plain")
        else:
            raise HTTPException(status_code=403, detail="Verification token mismatch")

    raise HTTPException(status_code=400, detail="Missing parameters")

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Endpoint to receive incoming webhook events from WhatsApp.
    """
    body = await request.json()
    
    # Verify the request is from a WhatsApp Business Account
    if body.get("object") == "whatsapp_business_account":
        try:
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    
                    # Log incoming messages
                    if "messages" in value:
                        for message in value["messages"]:
                            print(f"Received message: {message}")
                            
                    # Log status updates (delivered, read, etc.)
                    if "statuses" in value:
                        for status in value["statuses"]:
                            print(f"Received status update: {status}")
                            
            # Always return a 200 OK to acknowledge receipt of the event
            return Response(content="EVENT_RECEIVED", status_code=200)
            
        except Exception as e:
            print(f"Error processing webhook: {e}")
            return Response(content="ERROR", status_code=500)
    else:
        # Return a 404 for unrecognized events
        raise HTTPException(status_code=404, detail="Not a WhatsApp API event")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
