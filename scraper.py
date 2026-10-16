import re
import time
import gc
import logging
import requests
import pandas as pd
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

# Scrape ALL reviews for an AliExpress product using AliExpress feedback API.
class AliExpressReviewScraper:
    def __init__(self, user_agent: Optional[str] = None, pause_s: float = 1.0,
                 retries: int = 3, timeout: int = 15):
        self.base_url = "https://feedback.aliexpress.com/pc/searchEvaluation.do"
        self.pause_s = pause_s
        self.retries = retries
        self.timeout = timeout

        self.sess = requests.Session()
        self.headers = {
            "accept": "application/json, text/plain, */*",
            "user-agent": user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "accept-language": "en-US,en;q=0.9",
        }

    @staticmethod
    def extract_product_id(url: str) -> str:
        for pat in (r"/item/(\d+)\.html", r"/item/(\d+)", r"productId[=:](\d+)"):
            m = re.search(pat, url)
            if m:
                return m.group(1)
        raise ValueError("Could not extract product ID from URL")

    def _request(self, params: Dict[str, Any]):
        for attempt in range(1, self.retries + 1):
            try:
                r = self.sess.get(
                    self.base_url,
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    return r.json()
                log.warning("Attempt %d/%d: HTTP %s", attempt, self.retries, r.status_code)
            except (requests.RequestException, ValueError) as exc:
                log.warning("Attempt %d/%d failed: %s", attempt, self.retries, exc)
            time.sleep(self.pause_s)
        return None

    def scrape_all(self, product_id: str, page_size: int = 20, max_pages_cap: int = 10000):
        all_rows = []
        seen_ids = set()

        base_params = {
            "productId": product_id,
            "filter": "all",
            "sort": "complex_default",
            "pageSize": str(page_size),
        }

        page = 1

        while True:
            if page > max_pages_cap:
                break

            params = dict(base_params, page=str(page))
            data = self._request(params)

            if not data or "data" not in data:
                break

            d = data.get("data") or {}
            reviews = d.get("evaViewList") or []
            if not reviews:
                break

            for rv in reviews:
                rid = rv.get("evaluationId", "")
                if rid and rid in seen_ids:
                    continue

                seen_ids.add(rid)
                all_rows.append({
                    "review_id": rid,
                    "buyer_name": rv.get("buyerName", "Anonymous"),
                    "buyer_country": rv.get("buyerCountry", ""),
                    "rating": rv.get("buyerEval", None),
                    "review_text": rv.get("buyerFeedback", ""),
                    "review_date": rv.get("evalTime", "")
                })

            total_pages = d.get("totalPage", None)
            if total_pages and page >= int(total_pages):
                break

            gc.collect()
            time.sleep(self.pause_s)
            page += 1

        return pd.DataFrame(all_rows)
