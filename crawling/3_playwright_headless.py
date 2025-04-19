# scrape_coupang_jobs.py
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

def scrape_data_jobs(max_pages=50):
    """
    Coupang Jobs 'data' 검색 결과를 크롤링합니다.
    max_pages: 순회할 최대 페이지 수 (안정성을 위해 설정)
    """
    results = []
    base_url = 'https://www.coupang.jobs'
    search_path = '/en/jobs/?origin=global&search=data&page={}'

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for page_num in range(1, max_pages + 1):
            url = base_url + search_path.format(page_num)
            print(f"🕸  Fetching page {page_num}: {url}")
            page.goto(url, wait_until='networkidle')

            # gh_jid 파라미터를 가진 링크(<a>)만 추출
            job_links = page.query_selector_all('a[href*="?gh_jid="]')
            if not job_links:
                print("🔚 더 이상 공고가 없습니다. 종료합니다.")
                break

            for link in job_links:
                title = link.inner_text().strip()
                href = link.get_attribute('href')
                full_url = urljoin(base_url, href)

                # 공고 요소 바로 아래에 있을 수 있는 <p> 또는 <span>에서 location 추출
                loc_el = link.query_selector('p, span')
                location = loc_el.inner_text().strip() if loc_el else ''

                results.append({
                    'title': title,
                    'url': full_url,
                    'location': location
                })

            print(f"✅ Page {page_num}: {len(job_links)}개 수집")
        
        browser.close()
    return results

if __name__ == '__main__':
    jobs = scrape_data_jobs(max_pages=100)
    print(f"\n총 {len(jobs)}개의 공고를 크롤링했습니다.\n")
    for idx, job in enumerate(jobs, 1):
        print(f"{idx:3d}. {job['title']} | {job['location']} | {job['url']}")
