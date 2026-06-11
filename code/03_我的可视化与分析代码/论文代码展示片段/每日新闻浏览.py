async def fetch_all_news():
    # 抓取GNews
    async def fetch_gnews():
        params = {
            "token": settings.gnews_api_key,
            "lang": settings.gnews_lang,
            "max": settings.gnews_max_items,
        }
        async with httpx.AsyncClient(timeout=7.0, follow_redirects=True) as c:
            r = await c.get(settings.gnews_top_headlines_url, params=params)
        return r.json().get("articles", [])

    # 抓取单个RSS源
    async def fetch_rss(source, url):
        async with httpx.AsyncClient(timeout=7.0, follow_redirects=True) as c:
            r = await c.get(url)
        entries = feedparser.parse(r.content).entries[:30]
        return [
            {"source": source, "title": e.get("title"), "link": e.get("link")}
            for e in entries if e.get("title") and e.get("link")
        ]



    # 并发抓取：GNews + 多个RSS源
    gnews_task = fetch_gnews()
    rss_tasks = [fetch_rss(source, url) for source, url in RSS_SOURCES]
    gnews_articles, *rss_groups = await asyncio.gather(gnews_task, *rss_tasks)

    return gnews_articles, rss_groups

