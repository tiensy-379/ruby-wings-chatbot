# meta_param_builder.py
# Tạo lớp ParamBuilder trực tiếp trong file này
class ParamBuilder:
    def __init__(self):
        pass
    
    def build(self, request):
        # Logic đơn giản để trích xuất fbp/fbc
        return {"fbp": "", "fbc": ""}
import hashlib

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
