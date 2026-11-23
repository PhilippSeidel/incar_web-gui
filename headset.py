"""Headset pairing as the INCAR GUI does it. The QR code carries the core's WebRTC offer in compact form ("xra1:");
XR Commander (https://teleop.incar-robotics.se) on the headset answers through a small page this bridge serves on
the LAN, and the bridge hands the answer to the core over its signaling socket."""
import asyncio
import base64
import gzip
import html
import json
import socket

from aiohttp import web

BRIDGE_PORT = 8765  # the INCAR GUI's bridge port; XR Commander gets the full URL from the QR code
COMMANDER_ORIGIN = 'https://teleop.incar-robotics.se'


def lan_candidate(line: str) -> tuple[str, int] | None:
    """(ip, port) of an `a=candidate:` line for UDP over IPv4 of type host, else None."""
    if not line.startswith('a=candidate:'):
        return None
    parts = line[len('a=candidate:'):].split(' ')
    if len(parts) < 8 or parts[2].lower() != 'udp' or 'typ' not in parts or \
            parts[parts.index('typ') + 1:parts.index('typ') + 2] != ['host']:
        return None
    try:
        socket.inet_aton(parts[4])
        port = int(parts[5])
    except (OSError, ValueError):
        return None
    return (parts[4], port) if port > 0 and parts[4].count('.') == 3 else None


def compact_offer(sdp: str, answer_url: str) -> str:
    """The QR text: "xra1:" + base64url(gzip(JSON)) with where to answer and what XR Commander needs of the offer."""
    ip = port = ufrag = pwd = fingerprint = None
    c_ip, m_port = None, None
    for line in (raw.strip() for raw in sdp.replace('\r', '').split('\n')):
        if line.startswith('m=application '):
            m_port = int(line.split(' ')[1])
        elif line.startswith('c=IN IP4 '):
            c_ip = line[len('c=IN IP4 '):].strip()
        elif line.startswith('a=ice-ufrag:'):
            ufrag = line[len('a=ice-ufrag:'):].strip()
        elif line.startswith('a=ice-pwd:'):
            pwd = line[len('a=ice-pwd:'):].strip()
        elif line.startswith('a=fingerprint:sha-256 '):
            fingerprint = line[len('a=fingerprint:sha-256 '):].strip()
        elif ip is None and (candidate := lan_candidate(line)):
            ip, port = candidate
    ip, port = ip or c_ip, port or m_port
    if not (ip and port and ufrag and pwd and fingerprint):
        raise ValueError('The core offer lacks an address, ICE credentials or a sha-256 fingerprint')
    payload = json.dumps({'a': answer_url, 'o': {'ip': ip, 'p': port, 'u': ufrag, 'w': pwd, 'f': fingerprint}},
                         separators=(',', ':'))
    return 'xra1:' + base64.urlsafe_b64encode(gzip.compress(payload.encode())).decode().rstrip('=')


def answer_line(text: str) -> str:
    """The signaling line for the core from what XR Commander submits: JSON, or base64url of it."""
    text = text.strip()
    if not text.startswith(('{', '[')):
        text = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4)).decode()
    message = json.loads(text)
    if not isinstance(message, dict) or message.get('type') != 'answer' or not isinstance(message.get('sdp'), str):
        raise ValueError('Expected an SDP answer')
    return json.dumps({'sdp': message['sdp'], 'type': 'answer'})


def lan_ip_towards(ip: str) -> str:
    """This PC's address on the network that leads to ip (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('10.255.255.255' if ip.startswith('127.') else ip, 9))
        return s.getsockname()[0]


# what the INCAR bridge serves: checks it was opened by XR Commander, then posts the answer from the URL hash
SUBMIT_PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>XR Answer Receiver</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;font:16px/1.5 system-ui,sans-serif;
background:#101318;color:#fff}main{padding:24px;border-radius:20px;background:#1e2430;max-width:560px}
.error{color:#f87171}.ok{color:#4ade80}</style></head>
<body><main><h1>XR Answer Receiver</h1><p>Sends the XR answer to the incar web GUI on the local network.</p>
<p id="status">Waiting for answer payload...</p></main>
<script>
const origin = __ORIGIN__, status = document.getElementById('status');
function notify(ok, message, sessionId) {
  try { if (window.opener) window.opener.postMessage({type: 'xr-submit-result', ok, message, sessionId}, origin); } catch (_) {}
}
function handshake(sessionId) {
  return new Promise((resolve, reject) => {
    if (!window.opener) return reject(new Error('XR Commander opener window was not found.'));
    const token = Math.random().toString(36).slice(2);
    const timer = setTimeout(() => reject(new Error('XR Commander origin check failed.')), 1500);
    window.addEventListener('message', function listen(e) {
      const p = e.data;
      if (e.origin !== origin || !p || p.type !== 'xr-submit-ready' || p.token !== token || (p.sessionId || '') !== sessionId) return;
      clearTimeout(timer); window.removeEventListener('message', listen); resolve();
    });
    window.opener.postMessage({type: 'xr-submit-handshake', token, sessionId}, origin);
  });
}
(async () => {
  const params = new URLSearchParams(location.hash.slice(1));
  const answer = params.get('answer'), sessionId = params.get('sessionId') || '';
  try {
    if (!answer) throw new Error('No answer payload was found in the URL.');
    status.textContent = 'Verifying XR Commander origin...';
    await handshake(sessionId);
    status.textContent = 'Submitting answer...';
    const r = await fetch('/api/manual-answer', {method: 'POST', headers: {'Content-Type': 'text/plain; charset=utf-8'}, body: answer});
    if (!r.ok) throw new Error(await r.text() || 'Request failed with status ' + r.status);
    status.textContent = 'Answer submitted. Return to XR Commander.'; status.className = 'ok';
    notify(true, status.textContent, sessionId);
    setTimeout(() => { try { window.close(); } catch (_) {} }, 1200);
  } catch (e) {
    status.textContent = e.message || String(e); status.className = 'error';
    notify(false, status.textContent, sessionId);
  }
})();
</script></body></html>'''


class Bridge:
    """Holds the core's signaling socket from offer to answer and serves the answer page on the LAN."""

    def __init__(self):
        self.writer: asyncio.StreamWriter | None = None
        self.runner: web.AppRunner | None = None
        self.answered = False

    async def pair(self, core_ip: str, core_port: int) -> str:
        """Take the offer the core waits to send on its HMD port; returns the QR text."""
        await self.serve()
        self.release()
        for _ in range(40):  # the core may still be opening its signaling port
            try:
                reader, self.writer = await asyncio.open_connection(core_ip, core_port)
                break
            except OSError:
                await asyncio.sleep(0.25)
        else:
            raise ConnectionError(f'The core does not accept connections on {core_ip}:{core_port}')
        offer = json.loads(await asyncio.wait_for(reader.readline(), 10) or b'{}')
        if offer.get('type') != 'offer' or not isinstance(offer.get('sdp'), str):
            self.release()
            raise ConnectionError('The core sent no WebRTC offer')
        self.answered = False
        return compact_offer(offer['sdp'], f'http://{lan_ip_towards(core_ip)}:{BRIDGE_PORT}/xr-submit')

    async def submit(self, text: str) -> None:
        if self.writer is None or self.writer.is_closing() or self.answered:
            raise ValueError('No pairing is waiting for an answer: show the QR code in the GUI again')
        self.writer.write(answer_line(text).encode() + b'\n')
        await self.writer.drain()
        self.answered = True  # keep the socket: the core ends the headset session when it closes

    def release(self) -> None:
        """Close the signaling socket: the core then gives up the offer (or ends the headset session)."""
        if self.writer is not None:
            self.writer.close()
        self.writer = None

    async def serve(self) -> None:
        if self.runner is not None:
            return
        api = web.Application(client_max_size=256 * 1024)  # an SDP answer is a few kB
        api.router.add_get('/xr-submit', self.page)
        api.router.add_post('/xr-submit', self.form)
        api.router.add_post('/api/manual-answer', self.raw)
        api.router.add_get('/health', lambda _: web.json_response({'ok': True}))
        self.runner = web.AppRunner(api, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, '0.0.0.0', BRIDGE_PORT).start()  # the headset reaches it over the LAN

    async def stop(self) -> None:
        self.release()
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    async def page(self, _) -> web.Response:
        return web.Response(text=SUBMIT_PAGE.replace('__ORIGIN__', json.dumps(COMMANDER_ORIGIN)), content_type='text/html')

    async def accept(self, text: str) -> str | None:
        """None if the answer went to the core, else why not."""
        try:
            await self.submit(text)
        except (ValueError, OSError) as e:  # json and base64 errors are ValueErrors
            return str(e)
        return None

    async def raw(self, request: web.Request) -> web.Response:
        error = await self.accept(await request.text())
        return web.json_response({'ok': True} if error is None else {'ok': False, 'error': error},
                                 status=200 if error is None else 400)

    async def form(self, request: web.Request) -> web.Response:
        error = await self.accept(str((await request.post()).get('answer', '')))
        text = 'Answer submitted. Return to XR Commander.' if error is None else error
        return web.Response(text=f'<!doctype html><meta charset="utf-8"><p>{html.escape(text)}</p>',
                            content_type='text/html',
                            status=200 if error is None else 400)
