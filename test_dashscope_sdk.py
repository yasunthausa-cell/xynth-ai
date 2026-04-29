import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
import dashscope

dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

try:
    rsp = dashscope.ImageSynthesis.call(model=dashscope.ImageSynthesis.Models.wanx_v1,
                              prompt="A cute dog",
                              n=1,
                              size='1024*1024')
    if rsp.status_code == 200:
        print(rsp.output.results[0].url)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))
except Exception as e:
    print(f"Exception: {e}")
