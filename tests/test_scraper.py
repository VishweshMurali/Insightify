import pytest

from scraper import AliExpressReviewScraper


@pytest.mark.parametrize(
    "url",
    [
        "https://www.aliexpress.com/item/3256808746369608.html",
        "https://www.aliexpress.com/item/3256808746369608.html?spm=a2g0o",
        "https://aliexpress.us/item/3256808746369608",
        "https://example.com/page?productId=3256808746369608",
    ],
)
def test_extract_product_id(url):
    assert AliExpressReviewScraper.extract_product_id(url) == "3256808746369608"


def test_extract_product_id_rejects_unknown_urls():
    with pytest.raises(ValueError):
        AliExpressReviewScraper.extract_product_id("https://www.aliexpress.com/")


def _page(ids, total_pages):
    return {
        "data": {
            "totalPage": total_pages,
            "evaViewList": [
                {
                    "evaluationId": i,
                    "buyerName": "n",
                    "buyerCountry": "US",
                    "buyerEval": 80,
                    "buyerFeedback": f"text {i}",
                    "evalTime": "01 Jan 2024",
                }
                for i in ids
            ],
        }
    }


def _scraper(pages):
    s = AliExpressReviewScraper(pause_s=0)
    calls = []

    def fake_request(params):
        calls.append(int(params["page"]))
        return pages.get(int(params["page"]))

    s._request = fake_request
    return s, calls


def test_scrape_all_paginates_and_dedupes():
    s, calls = _scraper({1: _page(["a", "b"], 2), 2: _page(["b", "c"], 2)})
    df = s.scrape_all("1")
    assert list(df["review_id"]) == ["a", "b", "c"]
    assert calls == [1, 2]


def test_scrape_all_respects_page_cap():
    pages = {p: _page([f"r{p}"], 99) for p in range(1, 10)}
    s, calls = _scraper(pages)
    df = s.scrape_all("1", max_pages_cap=3)
    assert calls == [1, 2, 3]
    assert len(df) == 3


def test_scrape_all_stops_on_failed_request():
    s, calls = _scraper({1: _page(["a"], 5)})
    df = s.scrape_all("1")
    assert calls == [1, 2]
    assert list(df["review_id"]) == ["a"]
