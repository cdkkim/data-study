from firecrawl import FirecrawlApp
import pdb

api_key = ''
app = FirecrawlApp(api_key=api_key)

url = 'https://www.coupang.jobs/en/jobs/?search=data&location=&origin=global'

crawl_status = app.crawl_url(
  url, 
  params={
    'limit': 2, 
    'scrapeOptions': {'formats': ['markdown', 'html']}
  },
  poll_interval=30
)
pdb.set_trace()
print(crawl_status)
