"""
HTTP Stealth Module - Global stealth for all Python HTTP requests
Implements rotating user agents, headers, and anti-detection measures
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import aiohttp
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# USER AGENT ROTATION
# ============================================================================

USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "en-US,en;q=0.9,es;q=0.8",
    "en;q=0.9",
]

REFERERS = {
    "google": [
        "https://www.google.com/",
        "https://www.google.com/search?q=",
        "https://www.google.co.uk/",
    ],
    "bing": [
        "https://www.bing.com/",
        "https://www.bing.com/search?q=",
    ],
    "direct": [None],
}


@dataclass
class StealthSession:
    """Maintains consistent identity across requests"""
    user_agent: str = field(default_factory=lambda: random.choice(USER_AGENTS))
    accept_language: str = field(default_factory=lambda: random.choice(ACCEPT_LANGUAGES))
    request_count: int = 0
    last_request_time: float = 0
    blocked_count: int = 0
    current_engine: str = "google"  # google or bing
    
    def rotate_identity(self):
        """Rotate to new identity after block detection"""
        self.user_agent = random.choice(USER_AGENTS)
        self.accept_language = random.choice(ACCEPT_LANGUAGES)
        self.request_count = 0
        logger.info(f"[STEALTH] Rotated identity - new UA: {self.user_agent[:50]}...")
    
    def switch_engine(self):
        """Switch search engine after Google block"""
        if self.current_engine == "google":
            self.current_engine = "bing"
            logger.warning("[STEALTH] Google blocked - pivoting to Bing")
        else:
            self.current_engine = "google"
            logger.info("[STEALTH] Switching back to Google")
        self.blocked_count = 0
        self.rotate_identity()
    
    def get_headers(self, referer_type: str = "direct") -> Dict[str, str]:
        """Generate stealth headers"""
        referer_list = REFERERS.get(referer_type, REFERERS["direct"])
        referer = random.choice(referer_list)
        
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": self.accept_language,
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none" if referer is None else "cross-site",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }
        
        if referer:
            headers["Referer"] = referer
        
        return headers


# ============================================================================
# GLOBAL STEALTH CLIENT
# ============================================================================

class StealthHTTPClient:
    """
    Async HTTP client with global stealth measures
    - Randomized delays between requests
    - User agent rotation
    - Google -> Bing fallback
    - Block detection and recovery
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.stealth = StealthSession()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self.init_session()
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    async def init_session(self):
        """Initialize aiohttp session with stealth settings"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _add_human_delay(self):
        """Add randomized delay between requests"""
        # Base delay with jitter
        base_delay = random.uniform(0.5, 2.0)
        
        # Increase delay based on request count (fatigue simulation)
        fatigue_factor = min(self.stealth.request_count / 50, 1.0)
        delay = base_delay * (1 + fatigue_factor * 0.5)
        
        # Random longer pauses (reading/thinking simulation)
        if random.random() < 0.1:
            delay += random.uniform(2.0, 5.0)
        
        await asyncio.sleep(delay)
    
    def _is_blocked(self, status: int, text: str) -> bool:
        """Detect if request was blocked"""
        block_indicators = [
            "captcha",
            "unusual traffic",
            "automated queries",
            "rate limit",
            "blocked",
            "access denied",
            "too many requests",
            "verify you're human",
        ]
        
        if status in [429, 403, 503]:
            return True
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in block_indicators)
    
    async def get(
        self,
        url: str,
        referer_type: str = "direct",
        retry_on_block: bool = True
    ) -> Optional[str]:
        """
        Perform stealth GET request
        
        Args:
            url: Target URL
            referer_type: Type of referer to use (google, bing, direct)
            retry_on_block: Whether to retry with engine switch on block
        
        Returns:
            Response text or None if failed
        """
        await self.init_session()
        
        async with self._lock:
            await self._add_human_delay()
            self.stealth.request_count += 1
            self.stealth.last_request_time = time.time()
        
        headers = self.stealth.get_headers(referer_type)
        
        try:
            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                text = await response.text()
                
                if self._is_blocked(response.status, text):
                    logger.warning(f"[STEALTH] Block detected on {url[:50]}... (status={response.status})")
                    self.stealth.blocked_count += 1
                    
                    if self.stealth.blocked_count >= 2:
                        self.stealth.switch_engine()
                    else:
                        self.stealth.rotate_identity()
                    
                    if retry_on_block:
                        await asyncio.sleep(random.uniform(5, 10))
                        return await self.get(url, referer_type, retry_on_block=False)
                    return None
                
                return text
                
        except asyncio.TimeoutError:
            logger.error(f"[STEALTH] Timeout on {url[:50]}...")
            return None
        except Exception as e:
            logger.error(f"[STEALTH] Error on {url[:50]}...: {e}")
            return None
    
    def get_search_url(self, query: str) -> str:
        """Get search URL based on current engine"""
        encoded_query = query.replace(" ", "+")
        
        if self.stealth.current_engine == "google":
            return f"https://www.google.com/search?q={encoded_query}"
        else:
            return f"https://www.bing.com/search?q={encoded_query}"
    
    async def search(self, query: str) -> Optional[str]:
        """Perform search with current engine, fallback on block"""
        url = self.get_search_url(query)
        referer_type = self.stealth.current_engine
        
        result = await self.get(url, referer_type=referer_type)
        
        if result is None and self.stealth.current_engine == "google":
            # Try Bing as fallback
            self.stealth.switch_engine()
            url = self.get_search_url(query)
            result = await self.get(url, referer_type="bing", retry_on_block=False)
        
        return result


# ============================================================================
# GLOBAL CLIENT SINGLETON
# ============================================================================

_global_client: Optional[StealthHTTPClient] = None


async def get_stealth_client() -> StealthHTTPClient:
    """Get or create global stealth HTTP client"""
    global _global_client
    if _global_client is None:
        _global_client = StealthHTTPClient()
    await _global_client.init_session()
    return _global_client


async def close_stealth_client():
    """Close global stealth client"""
    global _global_client
    if _global_client:
        await _global_client.close()
        _global_client = None
