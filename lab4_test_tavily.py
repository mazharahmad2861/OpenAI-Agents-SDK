import os
from dotenv import load_dotenv
from tavily import TavilyClient 

load_dotenv()

tavily = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)

response = tavily.search(
    query="latest developments in AI agents",
    max_results=5
)

for result in response["results"]:
    print(result["title"])
    print(result["url"])
    print(result["content"])
    print("-" * 50)