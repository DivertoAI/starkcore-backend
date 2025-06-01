from fastapi import APIRouter, HTTPException, Request
import httpx
import os

router = APIRouter()

PAYPAL_CLIENT_ID = os.getenv("AVixfS2CXI5tJmqZ-zf9tnU2kTTSVMS1vZkOfxwb98r-JvBqYlViGo_qjQyL_LQPfZWQGnbSAdbq0uoD")
PAYPAL_CLIENT_SECRET = os.getenv("EAHYG9WmSfYvy-xD3Qd6TDAgUmpknz0pVLt0oG4UNQ_d1R6psfwZycPfopahLO8TI0a4Y6wSLO31NHce")
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com"
# "https://api-m.paypal.com"  # Use "https://api-m.sandbox.paypal.com" for sandbox

async def get_paypal_access_token():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{PAYPAL_API_BASE}/v1/oauth2/token",
            data={"grant_type": "client_credentials"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
        )
        response.raise_for_status()
        return response.json()["access_token"]

@router.post("/api/verify-paypal-payment")
async def verify_paypal_payment(request: Request):
    data = await request.json()
    order_id = data.get("order_id")

    if not order_id:
        raise HTTPException(status_code=400, detail="Order ID is required.")

    access_token = await get_paypal_access_token()

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{PAYPAL_API_BASE}/v2/checkout/orders/{order_id}",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid PayPal Order ID.")

        order_info = response.json()

        if order_info["status"] != "COMPLETED":
            raise HTTPException(status_code=400, detail="Payment not completed.")

        # OPTIONAL: You can check order_info["purchase_units"][0]["amount"]["value"] for amount match

        # ✅ Payment verified!
        # TODO: Add coins / subscription to the user here
        return {"message": "Payment verified successfully!", "details": order_info}