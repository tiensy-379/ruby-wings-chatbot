# meta_param_builder.py
from capi_param_builder import ParamBuilder

class MetaParamService:
    def __init__(self):
        # Khai báo ETLD+1 domain (PHẢI đúng domain chạy thật)
        self.builder = ParamBuilder([
            "rubywings.vn",
            "ruby-wings-chatbot.onrender.com"
        ])

    def process_request(self, request):
        """
        BẮT BUỘC gọi hàm này ở mỗi request có event Meta
        """
        host = request.host
        query_params = request.args.to_dict(flat=False)
        cookies = request.cookies
        referer = request.headers.get("Referer")
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        remote_addr = request.remote_addr

        # Process & build params
        self.builder.process_request(
            host,
            query_params,
            cookies,
            referer,
            x_forwarded_for,
            remote_addr
        )

        return self.builder.get_cookies_to_set()

    def get_fbc(self):
        return self.builder.get_fbc()

    def get_fbp(self):
        return self.builder.get_fbp()

    def get_client_ip(self):
        return self.builder.get_client_ip_address()

    def hash_pii(self, value, data_type):
        """
        data_type: email | phone | external_id | first_name | last_name ...
        """
        if not value:
            return None
        return self.builder.get_normalized_and_hashed_pii(value, data_type)
