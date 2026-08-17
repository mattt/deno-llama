/** Minimal self-contained chat UI for the desktop preview. */
export const UI = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>deno-llama</title>
<style>
  :root { color-scheme: light dark; font-family: -apple-system, system-ui, sans-serif; }
  body { margin: 0; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 12px 16px; font-weight: 600; border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  #log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { white-space: pre-wrap; line-height: 1.4; }
  .user { align-self: flex-end; background: color-mix(in srgb, currentColor 10%, transparent); padding: 8px 12px; border-radius: 12px; max-width: 75%; }
  .bot { align-self: flex-start; max-width: 90%; }
  form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  input { flex: 1; padding: 10px 12px; border-radius: 10px; border: 1px solid color-mix(in srgb, currentColor 25%, transparent); background: transparent; color: inherit; font: inherit; }
  button { padding: 10px 16px; border-radius: 10px; border: 0; background: #2563eb; color: white; font: inherit; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header>deno-llama · local chat</header>
<div id="log"></div>
<form id="f">
  <input id="i" placeholder="Ask something…" autocomplete="off" autofocus />
  <button id="b" type="submit">Send</button>
</form>
<script>
const log = document.getElementById("log");
const form = document.getElementById("f");
const input = document.getElementById("i");
const button = document.getElementById("b");

function add(cls, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  button.disabled = true;
  add("user", message);
  const bot = add("bot", "");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bot.textContent += dec.decode(value, { stream: true });
      log.scrollTop = log.scrollHeight;
    }
  } catch (err) {
    bot.textContent = "error: " + err;
  } finally {
    button.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>`;
