# meta_param_builder.py
from capi_param_builder import ParamBuilder


class MetaParamService:
    def __init__(self):
        # ETLD+1 domains PHẢI đúng domain thực tế
        self.builder = ParamBuilder([
            "rubywings.vn",
            "ruby-wings-chatbot.onrender.com"
        ])

    def process_request(self, request):
        """
        BẮT BUỘC gọi ở mỗi request có event Meta CAPI
        """
        host = request.host
        query_params = request.args.to_dict(flat=False)
        cookies = request.cookies
        referer = request.headers.get("Referer")

        # ✅ GỌI ĐÚNG API (CHỈ 4 THAM SỐ)
        self.builder.process_request(
            host,
            query_params,
            cookies,
            referer
        )

        # Trả về danh sách cookie cần set (_fbc, _fbp)
        return self.builder.get_cookies_to_set()

    def get_fbc(self):
        return self.builder.get_fbc()

    def get_fbp(self):
        return self.builder.get_fbp()

    def get_client_ip(self):
        # IP được builder tự chọn tốt nhất từ request
        return self.builder.get_client_ip_address()

    def hash_pii(self, value, data_type):
        """
        data_type: phone | email | external_id | first_name | last_name | ...
        """
        if not value:
            return None
        return self.builder.get_normalized_and_hashed_pii(value, data_type)
