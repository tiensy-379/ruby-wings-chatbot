"""
meta_capi.py - Server-side Meta Conversion API
Version: 2.0 (Enhanced with Call Button Tracking)

CHỨC NĂNG HIỆN CÓ:
1. send_meta_pageview: Gửi PageView event khi trang được load
2. send_meta_lead: Gửi Lead event khi có form submit
3. send_meta_call_button: Gửi Call Button event khi click nút gọi điện (FIXED)

CẬP NHẬT QUAN TRỌNG:
1. Hàm send_meta_call_button đã được fix để Meta ghi nhận đúng
2. Tương thích với tracking script nút gọi điện hiện tại
3. Giữ nguyên tất cả chức năng hiện có
"""

import time
import requests
import os
import uuid
import hashlib
import json
# Thêm vào dòng 1-3 của meta_capi.py:
import logging
logger = logging.getLogger("meta_capi")

# =========================
# EXISTING FUNCTIONS (GIỮ NGUYÊN)
# =========================

def send_meta_pageview(request):
    """
    Gửi PageView event đến Meta CAPI - KHÔNG THAY ĐỔI
    Chạy tự động khi trang được load
    """
    try:
        print("=== META CAPI HIT ===")

        pixel_id = os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")

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

        if test_event_code:
            payload["test_event_code"] = test_event_code

        url = f"https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}"
        response = requests.post(url, json=payload, timeout=5)
        logger.info(f"Meta CAPI response: {response.status_code} - {response.text}")

        

        print("META CAPI STATUS:", response.status_code)
        print("META CAPI RESPONSE TEXT:", response.text)
        print("META CAPI EVENT_ID:", event_id)

    except Exception as e:
        print("META CAPI EXCEPTION:", str(e))


def send_meta_lead(
    request,
    event_name="Lead",
    event_id=None,
    phone=None,
    value=None,
    currency="VND",
    content_name=None
):
    """
    Server-side Meta CAPI Lead Event - KHÔNG THAY ĐỔI
    Gửi khi có form submit, lead generation
    """
    # Feature flag: mặc định OFF
    if os.getenv("ENABLE_META_CAPI_LEAD", "false").lower() not in ("1", "true", "yes"):
        return

    try:
        pixel_id = os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")

        if not pixel_id or not token:
            return

        if not event_id:
            event_id = str(uuid.uuid4())

        user_data = {
            "client_ip_address": request.remote_addr,
            "client_user_agent": request.headers.get("User-Agent")
        }

        if phone:
            user_data["ph"] = _hash(phone)

        payload_event = {
            "event_name": event_name,
            "event_time": int(time.time()),
            "event_id": event_id,
            "event_source_url": request.url,
            "action_source": "website",
            "user_data": user_data
        }

        if value is not None:
            payload_event["custom_data"] = {
                "value": value,
                "currency": currency
            }
            if content_name:
                payload_event["custom_data"]["content_name"] = content_name

        payload = {"data": [payload_event]}

        if test_event_code:
            payload["test_event_code"] = test_event_code

        url = f"https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}"
        requests.post(url, json=payload, timeout=5)

    except Exception:
        # Fail-safe tuyệt đối
        return


# =========================
# ENHANCED FUNCTIONS (CẬP NHẬT CHO CALL BUTTON)
# =========================

def _hash(value: str) -> str:
    """Hash SHA256 cho PII data theo yêu cầu Meta"""
    if not value:
        return ""
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def send_meta_call_button(
    request,
    # === THAM SỐ TỪ CLIENT (QUAN TRỌNG) ===
    page_url: str = None,           # URL từ window.location.href (client)
    user_agent: str = None,         # User agent từ client
    # === THAM SỐ TỪ TRACKING SCRIPT ===
    phone: str = None,
    call_type: str = "phone",
    fbp: str = None,                # Từ cookie _fbp
    fbc: str = None,                # Từ cookie _fbc
    event_id: str = None,           # Event ID từ client
    pixel_id: str = None,           # Pixel ID từ client (có thể override)
    event_name: str = "CallButtonClick",
    value: float = 150000,
    currency: str = "VND",
    # === THAM SỐ TÙY CHỌN ===
    client_ip: str = None,          # IP từ client (nếu có proxy)
    content_name: str = "Call Button Click"
):
    """
    ENHANCED VERSION: Gửi Call Button event đến Meta CAPI
    TƯƠNG THÍCH với tracking script hiện tại của nút gọi điện
    
    Tham số được gửi từ tracking script:
    {
        phone: '0332510486',
        call_type: 'regular' hoặc 'zalo',
        page_url: window.location.href,
        user_agent: navigator.userAgent,
        fbp: getFBP(),        // từ cookie _fbp
        fbc: getFBC(),        // từ cookie _fbc hoặc fbclid
        event_id: generateEventID(),
        pixel_id: '862531473384426',  // pixel ID từ script
        event_name: 'CallButtonClick'
    }
    """
    
    # Feature flag
    if os.getenv("ENABLE_META_CAPI_CALL", "false").lower() not in ("1", "true", "yes"):
        print("META CAPI CALL: Feature flag OFF")
        return None
    
    try:
        print("=== META CAPI CALL BUTTON TRACKING ===")
        
        # 1. Lấy Pixel ID (ưu tiên từ client, fallback env)
        target_pixel_id = pixel_id or os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")
        
        if not target_pixel_id or not token:
            print("META CAPI CALL ERROR: Missing PIXEL_ID or TOKEN")
            return None
        
        # 2. Ưu tiên thông tin từ CLIENT thay vì server
        # Nếu page_url không có từ client, dùng từ server như cũ
        final_page_url = page_url or request.url
        final_user_agent = user_agent or request.headers.get("User-Agent", "")
        final_client_ip = client_ip or request.remote_addr
        
        # 3. Tạo event_id nếu client không gửi
        if not event_id:
            event_id = str(uuid.uuid4())
        
        # 4. Chuẩn bị user_data với ưu tiên từ client
        user_data = {
            "client_ip_address": final_client_ip,
            "client_user_agent": final_user_agent
        }
        
        # 5. Thêm fbp/fbc nếu có từ client (QUAN TRỌNG để match với pixel)
        if fbp:
            user_data["fbp"] = fbp
        if fbc:
            user_data["fbc"] = fbc
        
        # 6. Hash phone number nếu có (theo yêu cầu Meta)
        if phone:
            user_data["ph"] = _hash(str(phone))
            print(f"META CAPI CALL: Phone hashed: {phone[:4]}... -> {user_data['ph'][:10]}...")
        
        # 7. Tạo event payload - TƯƠNG THÍCH với tracking script
        payload_event = {
            "event_name": event_name,
            "event_time": int(time.time()),
            "event_id": event_id,
            "event_source_url": final_page_url,  # URL từ CLIENT
            "action_source": "website",
            "user_data": user_data,
            "custom_data": {
                "value": value,
                "currency": currency,
                "call_type": call_type,
                "content_name": content_name,
                "button_location": "fixed_bottom_left",
                "content_category": "Zalo Call" if call_type == "zalo" else "Phone Call"
            }
        }
        
        payload = {"data": [payload_event]}
        
        # 8. Test mode
        if test_event_code:
            payload["test_event_code"] = test_event_code
            print(f"META CAPI CALL TEST MODE: {test_event_code}")
        
        # 9. Gửi request đến Meta
        url = f"https://graph.facebook.com/v18.0/{target_pixel_id}/events?access_token={token}"
        
        # Log để debug
        print(f"META CAPI CALL DETAILS:")
        print(f"  Pixel: {target_pixel_id}")
        print(f"  Event: {event_name}")
        print(f"  Page URL: {final_page_url[:80]}...")
        print(f"  Call Type: {call_type}")
        print(f"  Event ID: {event_id}")
        print(f"  FBP: {fbp[:20] if fbp else 'None'}...")
        print(f"  FBC: {fbc[:20] if fbc else 'None'}...")
        
        response = requests.post(
            url, 
            json=payload, 
            timeout=5,
            headers={
                "Content-Type": "application/json",
                "User-Agent": final_user_agent
            }
        )
        
        # 10. Log kết quả
        print(f"META CAPI CALL RESPONSE: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"META CAPI CALL SUCCESS")
            if "events_received" in result and result["events_received"] > 0:
                print(f"  ✅ Event received by Meta")
            return result
        else:
            print(f"META CAPI CALL ERROR: {response.text}")
            return None
            
    except Exception as e:
        print(f"META CAPI CALL EXCEPTION: {str(e)}")
        return None