
# 抓取页面并解析
resp = client.get(url)
soup = BeautifulSoup(resp.text, "lxml")

# 抓取标题和正文
title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
content = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

og_img = soup.find("meta", attrs={"property": "og:image"})
if og_img and og_img.get("content"):
    image_url = urljoin(url, og_img["content"])
else:
    first_img = soup.find("img")
    image_url = urljoin(url, first_img.get("src", "")) if first_img else ""


