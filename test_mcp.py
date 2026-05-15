import asyncio
from mcp_client import VkusVillMCP

async def test():
    mcp = VkusVillMCP()
    
    # Тест 1: пустой запрос
    print("=== search('') ===")
    try:
        results = await mcp.search("")
        print(f"Найдено: {len(results)}")
        for i, p in enumerate(results[:3]):
            print(f"  {i+1}. {p['name']} — {p.get('price',{}).get('current','?')}₽")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест 2: конкретный запрос
    print("\n=== search('молоко 3.2% 900 мл') ===")
    results = await mcp.search("молоко 3.2% 900 мл")
    for i, p in enumerate(results[:5]):
        print(f"  {i+1}. {p['name']} — {p.get('price',{}).get('current','?')}₽")
    
    # Тест 3: поиск по xml_id (если знаем)
    print("\n=== какие поля возвращаются ===")
    if results:
        print(f"Ключи: {results[0].keys()}")

asyncio.run(test())
