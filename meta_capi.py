#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
META CONVERSION API - PHIÊN BẢN HOÀN THIỆN
Version: 3.2 Professional
Created: 2026-01-13
Author: Ruby Wings AI Team

MÔ TẢ: Server-side Meta Conversion API tracking với đầy đủ tính năng
- PageView tracking cho mọi request
- Lead tracking cho form submissions
- Call Button tracking cho hotline clicks
- Enhanced user data với PII hashing
- Retry mechanism với exponential backoff
- Connection pooling cho performance
- Comprehensive error handling

TÍCH HỢP: Hoàn toàn tương thích với Ruby Wings Chatbot v5.2
"""

import os
import time
import json
import logging
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import lru_cache, wraps
from dataclasses import dataclass, asdict
import threading

# HTTP client với connection pooling
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.error("Requests library not available")

# ==================== CONFIGURATION ====================
logger = logging.getLogger("meta-capi")

@dataclass
class MetaCAPIConfig:
    """Meta CAPI configuration"""
    pixel_id: str = ""
    access_token: str = ""
    api_version: str = "v18.0"
    endpoint: str = "https://graph.facebook.com"
    test_event_code: str = ""
    
    # Feature toggles
    enable_pageview: bool = True
    enable_lead: bool = True
    enable_call_button: bool = True
    
    # Performance settings
    timeout: int = 5
    max_retries: int = 2
    retry_backoff_factor: float = 0.5
    enable_connection_pooling: bool = True
    pool_connections: int = 10
    pool_maxsize: int = 10
    
    # Privacy settings
    hash_pii: bool = True
    hash_algorithm: str = "sha256"
    remove_ip_after_hash: bool = False
    
    # Debug settings
    debug_mode: bool = False
    log_payload: bool = False
    validate_events: bool = True
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            pixel_id=os.environ.get("META_PIXEL_ID", "").strip(),
            access_token=os.environ.get("META_CAPI_TOKEN", "").strip(),
            api_version=os.environ.get("META_API_VERSION", "v18.0"),
            test_event_code=os.environ.get("META_TEST_EVENT_CODE", "").strip(),
            enable_pageview=os.environ.get("ENABLE_META_CAPI_PAGEVIEW", "true").lower() == "true",
            enable_lead=os.environ.get("ENABLE_META_CAPI_LEAD", "true").lower() == "true",
            enable_call_button=os.environ.get("ENABLE_META_CAPI_CALL", "true").lower() == "true",
            debug_mode=os.environ.get("DEBUG_META_CAPI", "false").lower() == "true"
        )
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(self.pixel_id and self.access_token)

# Global configuration
config = MetaCAPIConfig.from_env()

# ==================== HTTP CLIENT ====================
class MetaCAPIHTTPClient:
    """HTTP client với retry mechanism và connection pooling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize HTTP client"""
        if not REQUESTS_AVAILABLE:
            self.session = None
            logger.error("Requests library not available")
            return
        
        # Create session với connection pooling
        self.session = requests.Session()
        
        if config.enable_connection_pooling:
            # Configure connection pooling
            adapter = HTTPAdapter(
                pool_connections=config.pool_connections,
                pool_maxsize=config.pool_maxsize,
                max_retries=Retry(
                    total=config.max_retries,
                    backoff_factor=config.retry_backoff_factor,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
            
            # Mount adapter cho cả HTTP và HTTPS
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'RubyWings-MetaCAPI/3.2',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info("✅ Meta CAPI HTTP client initialized")
    
    def post(self, url: str, payload: Dict, timeout: int = None) -> Optional[requests.Response]:
        """Send POST request với error handling"""
        if not self.session:
            logger.error("HTTP client not initialized")
            return None
        
        if timeout is None:
            timeout = config.timeout
        
        try:
            start_time = time.time()
            response = self.session.post(
                url,
                json=payload,
                timeout=timeout
            )
            elapsed = (time.time() - start_time) * 1000
            
            if config.debug_mode:
                logger.debug(f"Meta CAPI Request: {url}")
                logger.debug(f"Response: {response.status_code} in {elapsed:.0f}ms")
                if config.log_payload:
                    logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False)[:500]}...")
            
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Meta CAPI timeout after {timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Meta CAPI connection error")
            return None
        except Exception as e:
            logger.error(f"Meta CAPI request error: {str(e)}")
            return None
    
    def close(self):
        """Close HTTP session"""
        if self.session:
            self.session.close()
            logger.info("Meta CAPI HTTP client closed")

# Initialize HTTP client
http_client = MetaCAPIHTTPClient()

# ==================== EVENT BUILDERS ====================
class EventBuilder:
    """Builder pattern cho Meta CAPI events"""
    
    @staticmethod
    def create_base_event(
        event_name: str,
        event_time: Optional[int] = None,
        event_id: Optional[str] = None,
        event_source_url: Optional[str] = None,
        action_source: str = "website"
    ) -> Dict[str, Any]:
        """Create base event structure"""
        if event_time is None:
            event_time = int(time.time())
        
        if event_id is None:
            event_id = str(uuid.uuid4())
        
        event = {
            "event_name": event_name,
            "event_time": event_time,
            "event_id": event_id,
            "action_source": action_source
        }
        
        if event_source_url:
            event["event_source_url"] = event_source_url
        
        return event
    
    @staticmethod
    def add_user_data(
        event: Dict[str, Any],
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        fbp: Optional[str] = None,
        fbc: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add user data với PII hashing"""
        user_data = {}
        
        # Add IP và User-Agent
        if client_ip and not config.remove_ip_after_hash:
            user_data["client_ip_address"] = client_ip
        
        if user_agent:
            user_data["client_user_agent"] = user_agent
        
        # Hash PII data
        if phone and config.hash_pii:
            user_data["ph"] = EventBuilder._hash_value(phone, config.hash_algorithm)
        
        if email and config.hash_pii:
            user_data["em"] = EventBuilder._hash_value(email, config.hash_algorithm)
        
        # Add Facebook cookies
        if fbp:
            user_data["fbp"] = fbp
        
        if fbc:
            user_data["fbc"] = fbc
        
        if user_data:
            event["user_data"] = user_data
        
        return event
    
    @staticmethod
    def add_custom_data(
        event: Dict[str, Any],
        value: Optional[float] = None,
        currency: str = "VND",
        content_name: Optional[str] = None,
        content_category: Optional[str] = None,
        content_ids: Optional[List[str]] = None,
        content_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Add custom data to event"""
        custom_data = {}
        
        if value is not None:
            custom_data["value"] = value
            custom_data["currency"] = currency
        
        if content_name:
            custom_data["content_name"] = content_name
        
        if content_category:
            custom_data["content_category"] = content_category
        
        if content_ids:
            custom_data["content_ids"] = content_ids
        
        if content_type:
            custom_data["content_type"] = content_type
        
        # Add any additional custom data
        if kwargs:
            custom_data.update(kwargs)
        
        if custom_data:
            event["custom_data"] = custom_data
        
        return event
    
    @staticmethod
    def _hash_value(value: str, algorithm: str = "sha256") -> str:
        """Hash value theo Meta requirements"""
        if not value:
            return ""
        
        try:
            # Normalize: lowercase, trim whitespace
            normalized = value.lower().strip()
            
            if algorithm == "sha256":
                return hashlib.sha256(normalized.encode()).hexdigest()
            elif algorithm == "md5":
                return hashlib.md5(normalized.encode()).hexdigest()
            else:
                logger.warning(f"Unsupported hash algorithm: {algorithm}")
                return hashlib.sha256(normalized.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Hash error: {e}")
            return ""

# ==================== EVENT VALIDATOR ====================
class EventValidator:
    """Validate events before sending"""
    
    @staticmethod
    def validate_event(event: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate event structure"""
        errors = []
        
        # Required fields
        required_fields = ["event_name", "event_time", "event_id", "action_source"]
        for field in required_fields:
            if field not in event:
                errors.append(f"Missing required field: {field}")
        
        # Validate event_time
        if "event_time" in event:
            event_time = event["event_time"]
            if not isinstance(event_time, int):
                errors.append("event_time must be integer")
            elif event_time < 0:
                errors.append("event_time cannot be negative")
            # Check if event_time is not too far in future (max 7 days)
            elif event_time > int(time.time()) + (7 * 24 * 60 * 60):
                errors.append("event_time too far in future")
        
        # Validate action_source
        if "action_source" in event:
            valid_sources = ["website", "app", "phone_call", "chat", "physical_store"]
            if event["action_source"] not in valid_sources:
                errors.append(f"Invalid action_source: {event['action_source']}")
        
        # Validate user_data
        if "user_data" in event:
            user_data = event["user_data"]
            
            # Check for at least one identifier
            identifiers = ["em", "ph", "client_ip_address", "client_user_agent", "fbp", "fbc"]
            has_identifier = any(key in user_data for key in identifiers)
            
            if not has_identifier:
                errors.append("user_data must contain at least one identifier")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def sanitize_event(event: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event data"""
        # Create a copy
        sanitized = event.copy()
        
        # Remove empty fields
        keys_to_remove = []
        for key, value in sanitized.items():
            if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del sanitized[key]
        
        # Truncate long strings
        for key, value in sanitized.items():
            if isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:497] + "..."
        
        return sanitized

# ==================== MAIN CAPI FUNCTIONS ====================
def send_meta_pageview(request) -> Optional[Dict[str, Any]]:
    """
    Send PageView event to Meta CAPI
    
    Args:
        request: Flask request object
    
    Returns:
        Meta API response or None
    """
    if not config.enable_pageview:
        return None
    
    if not config.is_valid():
        logger.warning("Meta CAPI not configured")
        return None
    
    try:
        # Build event
        event = EventBuilder.create_base_event(
            event_name="PageView",
            event_source_url=request.url if hasattr(request, 'url') else "",
            action_source="website"
        )
        
        # Add user data
        EventBuilder.add_user_data(
            event,
            client_ip=request.remote_addr if hasattr(request, 'remote_addr') else None,
            user_agent=request.headers.get("User-Agent") if hasattr(request, 'headers') else None
        )
        
        # Validate event
        if config.validate_events:
            is_valid, errors = EventValidator.validate_event(event)
            if not is_valid:
                logger.error(f"Invalid PageView event: {errors}")
                return None
        
        # Sanitize
        event = EventValidator.sanitize_event(event)
        
        # Build payload
        payload = {"data": [event]}
        
        # Add test event code if in debug mode
        if config.test_event_code and config.debug_mode:
            payload["test_event_code"] = config.test_event_code
            logger.info(f"Meta CAPI PageView (TEST MODE): {event['event_id']}")
        
        # Send to Meta
        return _send_to_meta(payload)
        
    except Exception as e:
        logger.error(f"Meta CAPI PageView error: {str(e)}")
        return None

def send_meta_lead(
    request,
    event_name: str = "Lead",
    event_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    value: Optional[float] = None,
    currency: str = "VND",
    content_name: Optional[str] = None,
    contact_name: Optional[str] = None,
    lead_source: str = "Chatbot",
    lead_status: str = "New",
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Send Lead event to Meta CAPI
    
    Args:
        request: Flask request object
        event_name: Event name (default: "Lead")
        event_id: Custom event ID
        phone: Phone number
        email: Email address
        value: Lead value
        currency: Currency code
        content_name: Content name
        contact_name: Contact name
        lead_source: Lead source
        lead_status: Lead status
        **kwargs: Additional custom data
    
    Returns:
        Meta API response or None
    """
    if not config.enable_lead:
        return None
    
    if not config.is_valid():
        logger.warning("Meta CAPI not configured")
        return None
    
    try:
        # Build event
        event = EventBuilder.create_base_event(
            event_name=event_name,
            event_id=event_id,
            event_source_url=request.url if hasattr(request, 'url') else "",
            action_source="website"
        )
        
        # Add user data
        EventBuilder.add_user_data(
            event,
            client_ip=request.remote_addr if hasattr(request, 'remote_addr') else None,
            user_agent=request.headers.get("User-Agent") if hasattr(request, 'headers') else None,
            phone=phone,
            email=email
        )
        
        # Add custom data
        custom_data_kwargs = {
            "value": value,
            "currency": currency,
            "content_name": content_name,
            "lead_source": lead_source,
            "lead_status": lead_status
        }
        
        if contact_name:
            custom_data_kwargs["contact_name"] = contact_name
        
        # Add additional kwargs
        custom_data_kwargs.update(kwargs)
        
        EventBuilder.add_custom_data(event, **custom_data_kwargs)
        
        # Validate event
        if config.validate_events:
            is_valid, errors = EventValidator.validate_event(event)
            if not is_valid:
                logger.error(f"Invalid Lead event: {errors}")
                return None
        
        # Sanitize
        event = EventValidator.sanitize_event(event)
        
        # Build payload
        payload = {"data": [event]}
        
        # Add test event code if in debug mode
        if config.test_event_code and config.debug_mode:
            payload["test_event_code"] = config.test_event_code
            logger.info(f"Meta CAPI Lead (TEST MODE): {event['event_id']} - Phone: {phone[:4] if phone else 'N/A'}...")
        
        # Send to Meta
        response = _send_to_meta(payload)
        
        if response:
            logger.info(f"✅ Meta CAPI Lead sent: {event['event_id']} - Phone: {phone[:4] if phone else 'N/A'}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Meta CAPI Lead error: {str(e)}")
        return None

def send_meta_call_button(
    request,
    page_url: Optional[str] = None,
    phone: Optional[str] = None,
    call_type: str = "regular",
    button_location: str = "fixed_bottom_left",
    button_text: str = "Gọi ngay",
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Send Call Button event to Meta CAPI
    
    Args:
        request: Flask request object
        page_url: Page URL where button was clicked
        phone: Phone number called
        call_type: Type of call (regular, zalo, viber, etc.)
        button_location: Button location on page
        button_text: Button text
        **kwargs: Additional custom data
    
    Returns:
        Meta API response or None
    """
    if not config.enable_call_button:
        return None
    
    if not config.is_valid():
        logger.warning("Meta CAPI not configured")
        return None
    
    try:
        # Build event
        event = EventBuilder.create_base_event(
            event_name="Contact",
            event_source_url=page_url or (request.url if hasattr(request, 'url') else ""),
            action_source="website"
        )
        
        # Add user data
        EventBuilder.add_user_data(
            event,
            client_ip=request.remote_addr if hasattr(request, 'remote_addr') else None,
            user_agent=request.headers.get("User-Agent") if hasattr(request, 'headers') else None,
            phone=phone
        )
        
        # Add custom data
        custom_data = {
            "value": 150000,  # Estimated value of a call
            "currency": "VND",
            "call_type": call_type,
            "button_location": button_location,
            "button_text": button_text,
            "content_name": "Ruby Wings Hotline Call",
            "content_category": "Zalo Call" if call_type == "zalo" else "Phone Call",
            "business_name": "Ruby Wings Travel",
            "hotline_number": "0332510486"
        }
        
        # Add additional kwargs
        custom_data.update(kwargs)
        
        EventBuilder.add_custom_data(event, **custom_data)
        
        # Validate event
        if config.validate_events:
            is_valid, errors = EventValidator.validate_event(event)
            if not is_valid:
                logger.error(f"Invalid Call Button event: {errors}")
                return None
        
        # Sanitize
        event = EventValidator.sanitize_event(event)
        
        # Build payload
        payload = {"data": [event]}
        
        # Add test event code if in debug mode
        if config.test_event_code and config.debug_mode:
            payload["test_event_code"] = config.test_event_code
            logger.info(f"Meta CAPI Call Button (TEST MODE): {event['event_id']}")
        
        # Log event
        masked_phone = f"{phone[:4]}..." if phone else "N/A"
        logger.info(
            f"📞 Meta CAPI Call Button: Event={event['event_name']}, "
            f"Phone={masked_phone}, Type={call_type}"
        )
        
        # Send to Meta
        response = _send_to_meta(payload)
        
        if response:
            received = response.get('events_received', 0)
            if received > 0:
                logger.info(f"✅ Meta CAPI Call Button successful: {received} event(s) received")
        
        return response
        
    except Exception as e:
        logger.error(f"Meta CAPI Call Button error: {str(e)}")
        return None

def send_meta_custom_event(
    request,
    event_name: str,
    user_data: Optional[Dict] = None,
    custom_data: Optional[Dict] = None,
    event_id: Optional[str] = None,
    event_source_url: Optional[str] = None,
    action_source: str = "website",
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Send custom event to Meta CAPI
    
    Args:
        request: Flask request object
        event_name: Custom event name
        user_data: User data dictionary
        custom_data: Custom data dictionary
        event_id: Custom event ID
        event_source_url: Event source URL
        action_source: Action source
        **kwargs: Additional event fields
    
    Returns:
        Meta API response or None
    """
    if not config.is_valid():
        logger.warning("Meta CAPI not configured")
        return None
    
    try:
        # Build base event
        event = EventBuilder.create_base_event(
            event_name=event_name,
            event_id=event_id,
            event_source_url=event_source_url or (request.url if hasattr(request, 'url') else ""),
            action_source=action_source
        )
        
        # Add additional kwargs
        event.update(kwargs)
        
        # Add user data
        if user_data:
            if "user_data" not in event:
                event["user_data"] = {}
            event["user_data"].update(user_data)
        
        # Ensure basic user data
        if "user_data" not in event:
            EventBuilder.add_user_data(
                event,
                client_ip=request.remote_addr if hasattr(request, 'remote_addr') else None,
                user_agent=request.headers.get("User-Agent") if hasattr(request, 'headers') else None
            )
        
        # Add custom data
        if custom_data:
            EventBuilder.add_custom_data(event, **custom_data)
        
        # Validate event
        if config.validate_events:
            is_valid, errors = EventValidator.validate_event(event)
            if not is_valid:
                logger.error(f"Invalid custom event: {errors}")
                return None
        
        # Sanitize
        event = EventValidator.sanitize_event(event)
        
        # Build payload
        payload = {"data": [event]}
        
        # Add test event code if in debug mode
        if config.test_event_code and config.debug_mode:
            payload["test_event_code"] = config.test_event_code
            logger.info(f"Meta CAPI Custom Event (TEST MODE): {event['event_id']}")
        
        # Send to Meta
        return _send_to_meta(payload)
        
    except Exception as e:
        logger.error(f"Meta CAPI custom event error: {str(e)}")
        return None

def send_meta_bulk_events(
    request,
    events: List[Dict],
    test_mode: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Send multiple events in bulk
    
    Args:
        request: Flask request object
        events: List of event dictionaries
        test_mode: Enable test mode
    
    Returns:
        Meta API response or None
    """
    if not config.is_valid():
        logger.warning("Meta CAPI not configured")
        return None
    
    if not events:
        logger.warning("No events to send")
        return None
    
    try:
        # Prepare events
        prepared_events = []
        
        for event in events:
            # Ensure event has basic structure
            if "event_time" not in event:
                event["event_time"] = int(time.time())
            
            if "event_id" not in event:
                event["event_id"] = str(uuid.uuid4())
            
            if "action_source" not in event:
                event["action_source"] = "website"
            
            # Validate event
            if config.validate_events:
                is_valid, errors = EventValidator.validate_event(event)
                if not is_valid:
                    logger.error(f"Invalid event in bulk: {errors}")
                    continue
            
            # Sanitize
            sanitized = EventValidator.sanitize_event(event)
            prepared_events.append(sanitized)
        
        if not prepared_events:
            logger.warning("No valid events to send")
            return None
        
        # Build payload
        payload = {"data": prepared_events}
        
        # Add test event code
        if test_mode or (config.test_event_code and config.debug_mode):
            payload["test_event_code"] = config.test_event_code
            logger.info(f"Meta CAPI Bulk (TEST MODE): {len(prepared_events)} events")
        
        # Send to Meta
        response = _send_to_meta(payload)
        
        if response:
            received = response.get('events_received', 0)
            logger.info(f"✅ Meta CAPI Bulk sent: {received}/{len(prepared_events)} events received")
        
        return response
        
    except Exception as e:
        logger.error(f"Meta CAPI bulk events error: {str(e)}")
        return None

# ==================== HELPER FUNCTIONS ====================
def _send_to_meta(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Send payload to Meta CAPI
    
    Args:
        payload: Event payload
    
    Returns:
        Meta API response or None
    """
    if not config.is_valid():
        return None
    
    # Build URL
    url = f"{config.endpoint}/{config.api_version}/{config.pixel_id}/events"
    
    # Add access token
    url_with_token = f"{url}?access_token={config.access_token}"
    
    # Send request
    response = http_client.post(url_with_token, payload)
    
    if response is None:
        return None
    
    # Parse response
    if response.status_code == 200:
        try:
            result = response.json()
            
            if config.debug_mode:
                logger.debug(f"Meta CAPI Response: {json.dumps(result, indent=2)}")
            
            return result
        except json.JSONDecodeError:
            logger.error(f"Meta CAPI invalid JSON response: {response.text[:200]}")
            return None
    else:
        logger.error(f"Meta CAPI error {response.status_code}: {response.text[:200]}")
        return None

def check_meta_capi_health() -> Dict[str, Any]:
    """
    Check Meta CAPI health status
    
    Returns:
        Health status dictionary
    """
    health = {
        "status": "unknown",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "pixel_id_set": bool(config.pixel_id),
            "token_set": bool(config.access_token),
            "api_version": config.api_version,
            "endpoint": config.endpoint
        },
        "features": {
            "pageview": config.enable_pageview,
            "lead": config.enable_lead,
            "call_button": config.enable_call_button
        },
        "performance": {
            "connection_pooling": config.enable_connection_pooling,
            "max_retries": config.max_retries,
            "timeout": config.timeout
        }
    }
    
    # Determine overall status
    if not config.is_valid():
        health["status"] = "unconfigured"
    elif not REQUESTS_AVAILABLE:
        health["status"] = "missing_dependencies"
    else:
        # Try to make a test request
        test_url = f"{config.endpoint}/{config.api_version}/{config.pixel_id}"
        test_response = http_client.post(f"{test_url}?access_token={config.access_token}&fields=id", {})
        
        if test_response and test_response.status_code == 200:
            health["status"] = "healthy"
            health["test_response"] = "success"
        else:
            health["status"] = "unreachable"
            health["test_response"] = "failed"
    
    return health

def get_meta_capi_config() -> Dict[str, Any]:
    """
    Get Meta CAPI configuration (for debugging)
    
    Returns:
        Configuration dictionary
    """
    # Return safe configuration (without token)
    safe_config = {
        "pixel_id": config.pixel_id[:4] + "..." + config.pixel_id[-4:] if config.pixel_id else "",
        "api_version": config.api_version,
        "endpoint": config.endpoint,
        "test_mode": bool(config.test_event_code),
        "debug_mode": config.debug_mode,
        "hash_pii": config.hash_pii,
        "timeout": config.timeout
    }
    
    return safe_config

# ==================== CLEANUP ====================
def cleanup():
    """Cleanup resources"""
    http_client.close()
    logger.info("Meta CAPI cleanup completed")

# ==================== INITIALIZATION ====================
# Log initialization
logger.info("=" * 60)
logger.info("🚀 META CAPI v3.2 PROFESSIONAL INITIALIZED")
logger.info("=" * 60)
logger.info(f"📊 Pixel ID: {config.pixel_id[:6]}...{config.pixel_id[-4:] if len(config.pixel_id) > 10 else ''}")
logger.info(f"🔧 Features: PageView={config.enable_pageview}, Lead={config.enable_lead}, Call={config.enable_call_button}")
logger.info(f"⚡ Performance: Pooling={config.enable_connection_pooling}, Retries={config.max_retries}")
logger.info(f"🔒 Privacy: Hash PII={config.hash_pii}, Remove IP after hash={config.remove_ip_after_hash}")
logger.info("=" * 60)

# Register cleanup on exit
import atexit
atexit.register(cleanup)