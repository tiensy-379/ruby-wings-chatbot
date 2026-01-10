import time
import requests
import os
import uuid

def send_meta_pageview(request):
    try:
        print("=== META CAPI HIT ===")

        pixel_id = os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")  # BẮT BUỘC khi test

        if not pixel_id or not token:
            print("META CAPI ERROR: Missing PIXEL_ID or TOKEN")
            return

        event_id = str(uuid.uuid4())

        payload = {
            "data": [
                {
                    "event_name": "PageView",
                    "event_time": int(time.time()),
                    "event_id": event_id,
                    "event_source_url": request.url,
                    "action_source": "website",
                    "user_data": {
                        "client_ip_address": request.remote_addr,
                        "client_user_agent": request.headers.get("User-Agent")
                    }
                }
            ]
        }

        # Chỉ dùng khi test → giúp event HIỆN NGAY trong Events Manager
        if test_event_code:
            payload["test_event_code"] = test_event_code

        url = f"https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}"
        response = requests.post(url, json=payload, timeout=5)

        print("META CAPI STATUS:", response.status_code)
        print("META CAPI RESPONSE TEXT:", response.text)
        print("META CAPI EVENT_ID:", event_id)

    except Exception as e:
        print("META CAPI EXCEPTION:", str(e))
