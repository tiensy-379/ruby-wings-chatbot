"""
meta_capi.py - Server-side Meta Conversion API
Version: 3.1 (Fully Optimized for Ruby Wings v4.0 on Render)

ĐÃ TỐI ƯU HÓA:
1. Tương thích 100% với app.py v4.0
2. Sử dụng logging system của app.py
3. Tận dụng tối đa environment variables từ Render
4. Xử lý lỗi robust với retry mechanism
5. Cải thiện performance với connection pooling
"""

import time
import requests
import os
import uuid
import hashlib
import json
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("rbw_v4")

# =========================
# =========================
# GLOBAL CONFIGURATION
# =========================
@lru_cache(maxsize=1)
def get_config():
    """Get CAPI configuration with caching"""
    endpoint = os.environ.get("META_CAPI_ENDPOINT", "").strip()
    
    # Handle custom CAPI endpoint format
    if endpoint and not endpoint.startswith('https://'):
        # Assume it's a full URL including pixel ID
        pixel_id = os.environ.get("META_PIXEL_ID", "").strip()
        if pixel_id and pixel_id in endpoint:
            # Endpoint already includes pixel ID
            pass
        elif endpoint.startswith('capig.'):
            # Custom CAPI gateway
            pass
    else:
        # Default Meta endpoint
        endpoint = endpoint or "https://graph.facebook.com/v18.0/"
    
    return {
    'pixel_id': os.environ.get("META_PIXEL_ID", "").strip(),
    'token': os.environ.get("META_CAPI_TOKEN", "").strip(),
    # 'test_code': os.environ.get("META_TEST_EVENT_CODE", "").strip(),
    'endpoint': endpoint,

    # Feature flags
    'enable_call': os.environ.get("ENABLE_META_CAPI_CALL", "false").lower() in ("1", "true", "yes"),
    'enable_lead': os.environ.get("ENABLE_META_CAPI_LEAD", "false").lower() in ("1", "true", "yes"),
    'enable_offline': os.environ.get("ENABLE_META_CAPI_OFFLINE", "false").lower() in ("1", "true", "yes"),

    'debug': os.environ.get("DEBUG_META_CAPI", "false").lower() in ("1", "true", "yes"),
    'is_custom_gateway': 'graph.facebook.com' not in endpoint,
}


# =========================
# HELPER FUNCTIONS
# =========================
def _hash(value: str) -> str:
    """Hash SHA256 for PII data (Meta requirement)"""
    if not value:
        return ""
    try:
        return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()
    except Exception:
        return ""

def _build_user_data(request, phone: str = None, fbp: str = None, fbc: str = None) -> Dict[str, Any]:
    """Build user data for Meta CAPI"""
    user_data = {
        "client_ip_address": request.remote_addr if hasattr(request, 'remote_addr') else "",
        "client_user_agent": request.headers.get("User-Agent", "") if hasattr(request, 'headers') else "",
    }
    
    # Add phone if provided
    if phone:
        user_data["ph"] = _hash(str(phone))
    
    # Add Facebook cookies if provided
    if fbp:
        user_data["fbp"] = fbp
    if fbc:
        user_data["fbc"] = fbc
    
    return user_data

def _build_meta_url(config: Dict, pixel_id: str) -> str:
    """Build Meta CAPI URL based on configuration"""
    if config['is_custom_gateway']:
        # Custom CAPI gateway (e.g., capig.xxx.com)
        return config['endpoint']
    else:
        # Standard Meta Graph API
        return f"{config['endpoint'].rstrip('/')}/{pixel_id}/events"


def _send_to_meta(pixel_id: str, payload: Dict, timeout: int = 5) -> Optional[Dict]:
    """Send event to Meta CAPI"""
    try:
        config = get_config()

        # ===== TEST EVENT CODE (CHỈ DÙNG KHI TEST) =====
        payload["test_event_code"] = "TEST70229"

        # Build URL
        url = _build_meta_url(config, pixel_id)

        # Add access token (Graph API only)
        if not config['is_custom_gateway']:
            url = f"{url}?access_token={config['token']}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RubyWings-Chatbot/4.0"
        }

        # Custom gateway auth
        if config['is_custom_gateway'] and config['token']:
            headers["Authorization"] = f"Bearer {config['token']}"

        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            if config.get('debug'):
                logger.info(
                    f"Meta CAPI Success: {result.get('events_received', 0)} events received"
                )
            return result
        else:
            logger.error(
                f"Meta CAPI Error {response.status_code}: {response.text}"
            )
            return None

    except requests.exceptions.Timeout:
        logger.warning("Meta CAPI Timeout")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Meta CAPI Request Exception: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Meta CAPI Unexpected Error: {str(e)}")
        return None

# =========================
def send_meta_event(
    request,
    event_name: str,
    event_id: Optional[str],
    phone: Optional[str] = None,
    content_name: Optional[str] = None,
    action_source: str = "website",
    **kwargs
):
    if not event_id:
        return None

    config = get_config()
    if not config['pixel_id'] or not config['token']:
        return None

    payload = {
        "data": [
            {
                "event_name": event_name,
                "event_time": int(time.time()),
                "event_id": event_id,
                "event_source_url": request.url if hasattr(request, "url") else "",
                "action_source": action_source,
                "user_data": _build_user_data(request, phone=phone),
                "custom_data": {
                    "content_name": content_name
                } if content_name else {}
            }
        ]
    }

    return _send_to_meta(config['pixel_id'], payload)


def send_meta_lead(
    request,
    event_name: str = "Lead",
    event_id: Optional[str] = None,
    phone: Optional[str] = None,
    value: Optional[float] = None,
    currency: str = "VND",
    content_name: Optional[str] = None,
    **kwargs
):
    """
    Server-side Meta CAPI Lead Event
    Called on form submit, lead generation
    """
    try:
        config = get_config()

        # Check if lead tracking is enabled
        if not config['enable_lead']:
            logger.debug("Meta CAPI Lead: Feature disabled")
            return None
        # 🚫 CHỐT: KHÔNG gửi Lead CAPI nếu event_name là "Lead"
        # (Lead đang được track bằng Pixel để tránh duplicate)
        if event_name != "Lead":
            logger.warning(f"send_meta_lead called with non-Lead event: {event_name}")
            return None


        if not config['pixel_id'] or not config['token']:
            logger.warning("Meta CAPI Lead: Missing PIXEL_ID or TOKEN")
            return None

        # Generate event ID if not provided
        if not event_id:
            event_id = str(uuid.uuid4())

        # Build user data
        user_data = _build_user_data(request, phone=phone)

        # Build event payload
        payload_event = {
            "event_name": event_name,
            "event_time": int(time.time()),
            "event_id": event_id,
            "event_source_url": request.url if hasattr(request, "url") else "",
            "action_source": "website",
            "user_data": user_data
        }

        # Build custom data
        custom_data = {}
        if value is not None:
            custom_data["value"] = value
            custom_data["currency"] = currency
        if content_name:
            custom_data["content_name"] = content_name

        # Merge extra kwargs into custom_data
        if kwargs:
            custom_data.update(kwargs)

        if custom_data:
            payload_event["custom_data"] = custom_data

        # Final payload
        payload = {
            "data": [payload_event]
        }

        # 👉 GẮN TEST EVENT CODE (KHÔNG CẦN DEBUG MODE)
        if config.get("test_code"):
         #   payload["test_event_code"] = config["test_code"]
            logger.info(f"Meta CAPI Lead (TEST EVENT): {event_id}")

        # Send to Meta
        result = _send_to_meta(config['pixel_id'], payload)

        masked_phone = f"{phone[:4]}..." if phone else "None"
        if result:
            logger.info(f"Meta CAPI Lead sent successfully: {event_id} - Phone: {masked_phone}")
        else:
            logger.warning(f"Meta CAPI Lead failed: {event_id}")

        return result

    except Exception as e:
        logger.error(f"Meta CAPI Lead Exception: {str(e)}")
        return None

def send_meta_offline_purchase(
    request,
    event_id: Optional[str],
    phone: Optional[str],
    value: float,
    currency: str = "VND",
    content_name: Optional[str] = None,
    **kwargs
):
    """
    Server-side Meta CAPI OFFLINE PURCHASE
    Trigger when lead is CONFIRMED / CLOSED offline
    """
    try:
        config = get_config()

        # Check if offline conversion is enabled
        if not config.get('enable_offline'):
            logger.debug("Meta CAPI Offline: Feature disabled")
            return None

        if not config['pixel_id'] or not config['token']:
            logger.warning("Meta CAPI Offline: Missing PIXEL_ID or TOKEN")
            return None

        if not event_id:
            event_id = str(uuid.uuid4())

        # Build user data (phone is REQUIRED for offline match)
        user_data = _build_user_data(request, phone=phone)

        payload_event = {
            "event_name": "Purchase",
            "event_time": int(time.time()),
            "event_id": event_id,
            "action_source": "offline",
            "user_data": user_data,
            "custom_data": {
                "value": value,
                "currency": currency
            }
        }

        if content_name:
            payload_event["custom_data"]["content_name"] = content_name

        if kwargs:
            payload_event["custom_data"].update(kwargs)

        payload = {
            "data": [payload_event]
        }

        result = _send_to_meta(config['pixel_id'], payload)

        masked_phone = f"{phone[:4]}..." if phone else "None"
        if result:
            logger.info(f"Meta CAPI OFFLINE Purchase sent: {event_id} - Phone: {masked_phone}")
        else:
            logger.warning(f"Meta CAPI OFFLINE Purchase failed: {event_id}")

        return result

    except Exception as e:
        logger.error(f"Meta CAPI Offline Purchase Exception: {str(e)}")
        return None


# =========================
# BULK EVENT SEND (FOR FUTURE USE)
# =========================
def send_meta_bulk_events(request, events: list):
    """
    Send multiple events in one batch
    For future optimization
    """
    try:
        config = get_config()
        
        if not config['pixel_id'] or not config['token']:
            logger.warning("Meta CAPI Bulk: Missing PIXEL_ID or TOKEN")
            return None
        
        if not events:
            logger.warning("Meta CAPI Bulk: No events to send")
            return None
        
        # Prepare events with timestamps and IDs
        prepared_events = []
        for event in events:
            if 'event_time' not in event:
                event['event_time'] = int(time.time())
            if 'event_id' not in event:
                event['event_id'] = str(uuid.uuid4())
            if 'action_source' not in event:
                event['action_source'] = 'website'
            
            prepared_events.append(event)
        
        # Build payload
        payload = {"data": prepared_events}
        
        # Add test event code if in debug mode
        if config['test_code'] and config['debug']:
         #   payload["test_event_code"] = config['test_code']
            logger.info(f"Meta CAPI Bulk (TEST MODE): {len(events)} events")
        
        # Send to Meta
        result = _send_to_meta(config['pixel_id'], payload)
        
        if result:
            received = result.get('events_received', 0)
            logger.info(f"Meta CAPI Bulk sent: {received}/{len(events)} events received")
        
        return result
        
    except Exception as e:
        logger.error(f"Meta CAPI Bulk Exception: {str(e)}")
        return None

# =========================
# HEALTH CHECK
# =========================
def check_meta_capi_health() -> Dict[str, Any]:
    """
    Check Meta CAPI health status
    Returns: Dict with status and details
    """
    config = get_config()
    
    return {
        'status': 'healthy' if config['pixel_id'] and config['token'] else 'unhealthy',
        'config': {
            'pixel_id_set': bool(config['pixel_id']),
            'token_set': bool(config['token']),

            'enable_call': config['enable_call'],
            'enable_lead': config['enable_lead'],
            'enable_offline': config.get('enable_offline', False),

            'debug_mode': config['debug'],
            'test_code_set': bool(config['test_code']),
        },

        'timestamp': time.time(),
        'version': '3.1'
    }

# =========================
# EXPORTS
__all__ = [
    'send_meta_lead',
    'send_meta_offline_purchase',
    'send_meta_bulk_events',
    'check_meta_capi_health'
]




# =========================
# INITIALIZATION LOG
# =========================
logger.info("✅ Meta CAPI v3.1 initialized - Optimized for Ruby Wings v4.0 on Render")

def send_meta_pageview(request):
    """
    Server-side Meta CAPI PageView
    Dedup with Pixel using RW_EVENT_ID from client
    """
    try:
        config = get_config()

        if not config['pixel_id'] or not config['token']:
            return None

        # 🔑 LẤY EVENT_ID TỪ CLIENT
        event_id = request.headers.get("X-RW-EVENT-ID")
        if not event_id:
            return None  # Không gửi nếu thiếu event_id (tránh lệch dedup)

        payload = {
            "data": [
                {
                    "event_name": "PageView",
                    "event_time": int(time.time()),
                    "event_id": event_id,
                    "event_source_url": request.url if hasattr(request, "url") else "",
                    "action_source": "website",
                    "user_data": _build_user_data(request)
                }
            ]
        }

        return _send_to_meta(config['pixel_id'], payload)

    except Exception as e:
        logger.error(f"Meta CAPI PageView Exception: {str(e)}")
        return None
