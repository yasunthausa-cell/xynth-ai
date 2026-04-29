import urllib.parse, requests
prompt = "a beautiful landscape"
prompt_encoded = urllib.parse.quote(prompt)
url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width=1024&height=1024&nologo=true"
r = requests.get(url)
print(r.status_code)
print(url)
