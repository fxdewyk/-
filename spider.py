# spider.py (优化后的爬虫功能)
from flask import Blueprint, request, jsonify
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse, urljoin
import logging
import random
import time

spider_bp = Blueprint('spider', __name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 用户代理列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]


def validate_url(url):
    """验证 URL 格式"""
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme and parsed_url.netloc)


def clean_text(text):
    """清理文本，去除多余空白和特殊字符"""
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # 移除控制字符
    return text.strip()


@spider_bp.route('/api/spider/crawl', methods=['POST'])
def crawl_website():
    data = request.get_json()

    if not data or not data.get('url'):
        return jsonify({'success': False, 'message': '请提供URL'}), 400

    url = data['url']
    if not validate_url(url):
        return jsonify({'success': False, 'message': '无效的URL格式'}), 400

    try:
        # 设置请求头模拟浏览器访问
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }

        # 设置超时和重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"请求失败，重试中... ({attempt + 1}/{max_retries})")
                    time.sleep(1)
                else:
                    raise e

        # 检查内容类型
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return jsonify({
                'success': False,
                'message': f'不支持的内容类型: {content_type}'
            }), 400

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取标题
        title = soup.title.string.strip() if soup.title else "无标题"

        # 提取所有链接（去重和规范化）
        links_set = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if href and not href.startswith(('javascript:', 'mailto:', 'tel:')):
                absolute_url = urljoin(url, href)
                if absolute_url != url:  # 避免包含自身
                    links_set.add(absolute_url)
        links = list(links_set)[:50]  # 限制数量

        # 提取所有图片（去重）
        images_set = set()
        for img in soup.find_all('img', src=True):
            src = img['src'].strip()
            if src:
                absolute_src = urljoin(url, src)
                images_set.add(absolute_src)
        images = list(images_set)[:20]  # 限制数量

        # 提取文本内容
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        cleaned_text = clean_text(text)
        text_sample = cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text

        # 提取元描述
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'].strip() if meta_desc else "无描述"

        return jsonify({
            'success': True,
            'data': {
                'title': title,
                'description': description,
                'links': links,
                'images': images,
                'text_sample': text_sample,
                'status_code': response.status_code,
                'content_type': content_type,
                'url': url
            }
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"请求异常: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'抓取失败: {str(e)}'
        }), 500
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'处理数据时出错: {str(e)}'
        }), 500