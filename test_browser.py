# test_browser.py
import asyncio
from app.tool.browser_use_tool import BrowserUseTool

async def test_browser_use():
    browser_tool = BrowserUseTool()
    
    try:
        print("Testing navigation...")
        result = await browser_tool.execute({
            "action": "navigate",
            "url": "https://example.com"
        })
        print(f"Navigation result: {result.output}")
        
        print("\nTesting get_text...")
        result = await browser_tool.execute({
            "action": "get_text"
        })
        print(f"Text result (first 100 chars): {result.output[:100]}...")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        print("Cleaning up browser...")
        await browser_tool.cleanup()

if __name__ == "__main__":
    asyncio.run(test_browser_use())