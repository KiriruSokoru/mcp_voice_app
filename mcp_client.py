"""
mcp_client.py — клиент MCP-сервера ВкусВилла.

pip install mcp httpx
"""
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://mcp001.vkusvill.ru/mcp"

import re

def _clean(text: str) -> str:
    """Убирает HTML-entities типа &nbsp;"""
    return re.sub(r'&\w+;', ' ', text).strip()


async def _call(tool: str, args: dict) -> str:
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool(tool, args)
            for block in result.content:
                if hasattr(block, "text"):
                    return block.text
    raise ValueError(f"Нет ответа от {tool}")


class VkusVillMCP:

    async def search(self, query: str, page: int = 1, sort: str = "popularity") -> list:
        """Возвращает список товаров. Ключ: data.items"""
        raw = await _call("vkusvill_products_search",
                          {"q": query, "page": page, "sort": sort})
        data = json.loads(raw)
        items = data.get("data", {}).get("items", [])
        # чистим HTML-entities в названиях
        for item in items:
            item["name"] = _clean(item.get("name", ""))
        return items

    async def details(self, product_id: int) -> dict:
        """Детали товара по id (КБЖУ, состав)."""
        raw = await _call("vkusvill_product_details", {"id": int(product_id)})
        return json.loads(raw).get("data", {})

    async def cart_link(self, items: list) -> str:
        """
        items = [{"xml_id": int, "q": float}, ...]
        Возвращает URL share_basket.
        """
        if len(items) > 30:
            raise ValueError(f"Максимум 30 позиций, передано {len(items)}")
        raw = await _call("vkusvill_cart_link_create", {"products": items})
        data = json.loads(raw)
        return data.get("data", {}).get("link", raw)


# ── быстрая проверка ──────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def smoke():
        mcp = VkusVillMCP()

        print("=== search('молоко') ===")
        items = await mcp.search("молоко")
        for p in items[:3]:
            print(f"  [{p['xml_id']}] {p['name']} — {p.get('price',{}).get('current','?')}₽")

        first_id = items[0]["xml_id"]
        print(f"\n=== cart_link([{first_id}]) ===")
        url = await mcp.cart_link([{"xml_id": first_id, "q": 1}])
        print(f"  {url}")

    asyncio.run(smoke())
