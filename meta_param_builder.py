"""
Meta Parameter Builder Service
Extracts and validates Meta Pixel parameters (fbp, fbc) from requests
"""

import hashlib

class ParamBuilder:
    def __init__(self, config_list=None):
        self.config_list = config_list or []
        self.fbp = ''
        self.fbc = ''
        self.cookies = {}
    
    def process_request(self, host, args, cookies, referer):
        """
        Process request to extract fbc/fbp from args or cookies
        """
        # Extract from args (URL parameters)
        fbp_list = args.get('fbp', [])
        self.fbp = fbp_list[0] if fbp_list else cookies.get('fbp', '')
        
        fbc_list = args.get('fbc', [])
        self.fbc = fbc_list[0] if fbc_list else cookies.get('fbc', '')
        
        # Store cookies if needed
        if self.fbp:
            self.cookies['fbp'] = self.fbp
        if self.fbc:
            self.cookies['fbc'] = self.fbc
        
        return self
    
    def get_cookies_to_set(self):
        """Return cookies that should be set"""
        return self.cookies
    
    def get_fbc(self):
        """Get fbc value"""
        return self.fbc
    
    def get_fbp(self):
        """Get fbp value"""
        return self.fbp
    
    def build(self, request):
        """Alternative method to extract parameters (for compatibility)"""
        fbp = request.args.get('fbp') or request.headers.get('fbp') or ''
        fbc = request.args.get('fbc') or request.headers.get('fbc') or ''
        return {"fbp": fbp, "fbc": fbc}


class MetaParamService:
    def __init__(self):
        self.builder = ParamBuilder([
            "rubywings.vn",
            "ruby-wings-chatbot-v4.onrender.com"
        ])

    def process_request(self, request):
        """
        Chỉ dùng để xử lý fbc / fbp từ request
        """
        self.builder.process_request(
            request.host,
            request.args.to_dict(flat=False),
            request.cookies,
            request.headers.get("Referer"),
        )

        return self.builder.get_cookies_to_set()

    def get_fbc(self):
        return self.builder.get_fbc()

    def get_fbp(self):
        return self.builder.get_fbp()

    def get_client_ip(self, request):
        """
        IP phải tự lấy – SDK KHÔNG cung cấp
        """
        return (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.remote_addr
        )

    def hash_pii(self, value: str):
        """
        Hash SHA256 chuẩn Meta
        """
        if not value:
            return None
        v = value.strip().lower().encode("utf-8")
        return hashlib.sha256(v).hexdigest()