import os, requests, time

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    # try loading from .env
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    api_key = os.environ.get("DASHSCOPE_API_KEY")

def test_image(model_name):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "input": {"prompt": "a beautiful landscape"},
        "parameters": {"size": "1024*1024", "n": 1}
    }
    
    r = requests.post("https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis", headers=headers, json=payload, timeout=60)
    print(f"{model_name} POST response: {r.status_code} {r.text}")
    
    if r.status_code == 200:
        task_id = r.json().get("output", {}).get("task_id")
        print(f"Task ID: {task_id}")
        for _ in range(5):
            time.sleep(2)
            status_req = requests.get(f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            status_json = status_req.json()
            status = status_json.get("output", {}).get("task_status")
            print(f"Status: {status}")
            if status in ["SUCCEEDED", "FAILED"]:
                print(status_json)
                break

test_image("wanx2.1-t2i-turbo")
test_image("wanx-v1")
