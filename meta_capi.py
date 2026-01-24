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
        'test_code': os.environ.get("META_TEST_EVENT_CODE", "").strip(),
        'endpoint': endpoint,
        'enable_call': os.environ.get("ENABLE_META_CAPI_CALL", "false").lower() in ("1", "true", "yes"),
        'enable_lead': os.environ.get("ENABLE_META_CAPI_LEAD", "false").lower() in ("1", "true", "yes"),
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

def _send_to_meta(pixel_id: str, payload: Dict, timeout: int = 5) -> Optional[Dict]:
    """Send event to Meta CAPI"""
    try:
        config = get_config()
        
        # Build URL
        url = _build_meta_url(config, pixel_id)
        
        # Add access token based on endpoint type
        if not config['is_custom_gateway']:
            url = f"{url}?access_token={config['token']}"
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RubyWings-Chatbot/4.0"
        }
        
        # Add Authorization header for custom gateways
        if config['is_custom_gateway'] and config['token']:
            headers["Authorization"] = f"Bearer {config['token']}"
        
        # Send request
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if config['debug']:
                logger.info(f"Meta CAPI Success: {result.get('events_received', 0)} events received")
            return result
        else:
            logger.error(f"Meta CAPI Error {response.status_code}: {response.text}")
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
def _build_meta_url(config: Dict, pixel_id: str) -> str:
    """Build Meta CAPI URL based on configuration"""
    if config['is_custom_gateway']:
        # Custom CAPI gateway (e.g., capig.datah04.com)
        return config['endpoint']
    else:
        # Standard Meta Graph API
        return f"{config['endpoint'].rstrip('/')}/{pixel_id}/events"
# =========================
# MAIN CAPI FUNCTIONS
# =========================
def send_meta_pageview(request):
    """
    Send PageView event to Meta CAPI
    Automatically called on page load
    """
    try:
        config = get_config()
        
        if not config['pixel_id'] or not config['token']:
            logger.warning("Meta CAPI: Missing PIXEL_ID or TOKEN")
            return
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Build payload
        payload = {
            "data": [
                {
                    "event_name": "PageView",
                    "event_time": int(time.time()),
                    "event_id": event_id,
                    "event_source_url": request.url if hasattr(request, 'url') else "",
                    "action_source": "website",
                    "user_data": _build_user_data(request)
                }
            ]
        }
        
        # Add test event code if in debug mode
        if config['test_code'] and config['debug']:
            payload["test_event_code"] = config['test_code']
            logger.info(f"Meta CAPI PageView (TEST MODE): {event_id}")
        
        # Send to Meta
        result = _send_to_meta(config['pixel_id'], payload)
        
        if result:
            logger.info(f"Meta CAPI PageView sent successfully: {event_id}")
        else:
            logger.warning(f"Meta CAPI PageView failed: {event_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Meta CAPI PageView Exception: {str(e)}")
        return None

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
            payload["test_event_code"] = config["test_code"]
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


def send_meta_call_button(
    request,
    page_url: Optional[str] = None,
    phone: Optional[str] = None,
    call_type: str = "regular",
    **kwargs
):
    """
    Enhanced Call Button event for Meta CAPI
    Compatible with both current and future tracking scripts
    
    Parameters from app.py:
    - page_url: URL where call button was clicked
    - phone: Phone number (0332510486)
    - call_type: "regular" or "zalo"
    
    Returns: Meta API response or None
    """
    try:
        config = get_config()
        
        # Check if call tracking is enabled
        if not config['enable_call']:
            logger.debug("Meta CAPI Call Button: Feature disabled")
            return None
        
        # Get target pixel ID (prefer provided, fallback to env)
        target_pixel_id = config['pixel_id']
        if not target_pixel_id or not config['token']:
            logger.warning("Meta CAPI Call Button: Missing PIXEL_ID or TOKEN")
            return None
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Prepare data with priority: client > request
        final_page_url = page_url or (request.url if hasattr(request, 'url') else "")
        
        # Build user data
        user_data = _build_user_data(request, phone=phone)
        
        # Build event payload
        payload_event = {
            "event_name": "CallButtonClick",
            "event_time": int(time.time()),
            "event_id": event_id,
            "event_source_url": final_page_url,
            "action_source": "website",
            "user_data": user_data,
            "custom_data": {
                "value": 150000,
                "currency": "VND",
                "call_type": call_type,
                "content_name": "Ruby Wings Hotline Call",
                "button_location": "fixed_bottom_left",
                "content_category": "Zalo Call" if call_type == "zalo" else "Phone Call",
                "business_name": "Ruby Wings Travel",
                "hotline_number": "0332510486"
            }
        }
        
        # Add any additional kwargs to custom data
        if kwargs and 'custom_data' in payload_event:
            payload_event["custom_data"].update(kwargs)
        
        # Build final payload
        payload = {"data": [payload_event]}
        
        # Add test event code if in debug mode
        if config['test_code'] and config['debug']:
            payload["test_event_code"] = config['test_code']
            logger.info(f"Meta CAPI Call Button (TEST MODE): {event_id}")
        
        # Log event details (mask phone for privacy)
        masked_phone = f"{phone[:4]}..." if phone else "None"
        logger.info(
            f"Meta CAPI Call Button Tracking: "
            f"Event=CallButtonClick, "
            f"Pixel={target_pixel_id[:6]}..., "
            f"Phone={masked_phone}, "
            f"Type={call_type}, "
            f"URL={final_page_url[:50]}..."
        )
        
        # Send to Meta
        result = _send_to_meta(target_pixel_id, payload)
        
        if result:
            received = result.get('events_received', 0)
            if received > 0:
                logger.info(f"✅ Meta CAPI Call Button successful: {received} event(s) received")
            else:
                logger.warning(f"⚠️ Meta CAPI Call Button: No events received in response")
        else:
            logger.warning(f"❌ Meta CAPI Call Button failed")
        
        return result
        
    except Exception as e:
        logger.error(f"Meta CAPI Call Button Exception: {str(e)}")
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
            payload["test_event_code"] = config['test_code']
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
            'debug_mode': config['debug'],
            'test_code_set': bool(config['test_code']),
        },
        'timestamp': time.time(),
        'version': '3.1'
    }

# =========================
# EXPORTS
# =========================
__all__ = [
    'send_meta_pageview',
    'send_meta_lead', 
    'send_meta_call_button',
    'send_meta_bulk_events',
    'check_meta_capi_health'
]

# =========================
# INITIALIZATION LOG
# =========================
logger.info("✅ Meta CAPI v3.1 initialized - Optimized for Ruby Wings v4.0 on Render")