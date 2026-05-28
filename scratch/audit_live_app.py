import asyncio
import websockets
import json
import urllib.request
import subprocess
import time
import os
import base64

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
live_url = "https://ml-tying-the-data-knot-swipeiq-app.streamlit.app/"
output_dir = r"C:\Users\HP\.gemini\antigravity\brain\b0df43fa-5511-4dee-a9e5-9e34688919ec"

async def send_cdp(ws, method, params=None):
    if params is None:
        params = {}
    msg_id = int(time.time() * 1000)
    payload = {
        "id": msg_id,
        "method": method,
        "params": params
    }
    await ws.send(json.dumps(payload))
    while True:
        resp_raw = await ws.recv()
        resp = json.loads(resp_raw)
        if resp.get("id") == msg_id:
            return resp.get("result")

async def run_audit():
    print("Launching Chrome on port 9333...")
    chrome_proc = subprocess.Popen([
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--remote-debugging-port=9333",
        "--window-size=1280,1024",
        "about:blank"
    ])
    
    time.sleep(3)
    
    try:
        # Get websocket debugger URL
        with urllib.request.urlopen("http://127.0.0.1:9333/json/list") as response:
            targets = json.loads(response.read().decode())
            page_target = None
            for target in targets:
                if target.get("type") == "page":
                    page_target = target
                    break
            
            if not page_target:
                print("No page target found in Chrome!")
                return
            
            ws_url = page_target["webSocketDebuggerUrl"]
            print(f"Connecting to CDP websocket: {ws_url}")
            
            async with websockets.connect(ws_url) as ws:
                # Enable domains
                await send_cdp(ws, "Page.enable")
                await send_cdp(ws, "Runtime.enable")
                
                # Set viewport size
                await send_cdp(ws, "Emulation.setDeviceMetricsOverride", {
                    "width": 1280,
                    "height": 1024,
                    "deviceScaleFactor": 1,
                    "mobile": False
                })
                
                print(f"Navigating to live app: {live_url}")
                await send_cdp(ws, "Page.navigate", {"url": live_url})
                
                print("Waiting 12 seconds for the Streamlit app to load...")
                await asyncio.sleep(12)
                
                # Query page structure inside iframe if it exists
                check_js = """
                (function() {
                    const iframe = document.querySelector('iframe[id="streamlitApp"]');
                    if (!iframe) {
                        return { error: 'Streamlit iframe not found' };
                    }
                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                    if (!doc) {
                        return { error: 'Iframe document not accessible' };
                    }
                    
                    const header = doc.querySelector('.website-header');
                    if (!header) {
                        return { error: '.website-header element not found inside iframe' };
                    }
                    
                    const computedStyle = iframe.contentWindow.getComputedStyle(header);
                    const paddingLeft = computedStyle.paddingLeft;
                    
                    const logoBox = doc.querySelector('.web-logo-box');
                    const logoStyle = logoBox ? iframe.contentWindow.getComputedStyle(logoBox) : null;
                    const logoMarginLeft = logoStyle ? logoStyle.marginLeft : 'N/A';
                    
                    const dropdowns = Array.from(doc.querySelectorAll('.web-dropdown')).map(d => {
                        const label = d.querySelector('label');
                        const check = d.querySelector('input[type="checkbox"]');
                        const content = d.querySelector('.web-dropdown-content');
                        const contentStyle = content ? iframe.contentWindow.getComputedStyle(content) : null;
                        return {
                            labelText: label ? label.textContent.trim() : 'N/A',
                            hasCheckbox: !!check,
                            checkboxId: check ? check.id : 'N/A',
                            contentTop: contentStyle ? contentStyle.top : 'N/A'
                        };
                    });
                    
                    const links = Array.from(doc.querySelectorAll('.web-nav-menu a')).map(a => {
                        return {
                            text: a.textContent.trim(),
                            href: a.getAttribute('href'),
                            target: a.getAttribute('target')
                        };
                    });
                    
                    return {
                        success: true,
                        paddingLeft: paddingLeft,
                        logoMarginLeft: logoMarginLeft,
                        dropdowns: dropdowns,
                        links: links
                    };
                })()
                """
                
                # Wait a bit more to ensure iframe loads fully
                print("Checking elements...")
                eval_res = await send_cdp(ws, "Runtime.evaluate", {
                    "expression": check_js,
                    "returnByValue": True
                })
                
                result_val = eval_res.get("result", {}).get("value")
                print("Audit results from page:")
                print(json.dumps(result_val, indent=2))
                
                # Capture screenshot
                print("Capturing screenshot...")
                screenshot_res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"})
                if screenshot_res and "data" in screenshot_res:
                    img_data = base64.b64decode(screenshot_res["data"])
                    img_path = os.path.join(output_dir, "live_audit_snapshot.png")
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    print(f"Saved screenshot to: {img_path} ({len(img_data)} bytes)")
                else:
                    print("Failed to capture screenshot")
                    
    except Exception as e:
        print(f"Audit failed: {e}")
    finally:
        print("Stopping Chrome...")
        chrome_proc.terminate()
        chrome_proc.wait()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(run_audit())
