/* ─────────────────────────────────────────────────────────────
   AgriBot — Frontend Application Logic
   • JWT auth (register / login / logout)
   • Send messages → POST /chat → render bot reply
   • User messages on RIGHT, bot messages on LEFT
   • Username shown on every user message
   • Copy button on all messages
   • Edit button on user messages (opens edit overlay)
   • Sidebar open/close toggle
   • Chat history sidebar
   • Welcome screen with suggestion chips
   • Auto-resizing textarea
────────────────────────────────────────────────────────────── */

const API = 'https://agribot-pjdo.onrender.com';
const STORE  = 'agribot_token';
const USTORE = 'agribot_username';

/* ── State ──────────────────────────────────────────── */
let token          = localStorage.getItem(STORE)  || null;
let currentUser    = localStorage.getItem(USTORE) || null;
let sidebarOpen    = true;
let editingMsgEl   = null;   // reference to bubble being edited
let hasMessages    = false;  // track whether welcome screen should hide

/* ── DOM refs ───────────────────────────────────────── */
const authOverlay    = document.getElementById('auth-overlay');
const appContainer   = document.getElementById('app-container');
const authError      = document.getElementById('auth-error');
const authBtn        = document.getElementById('auth-btn');
const msgsEl         = document.getElementById('chat-messages');
const welcomeScreen  = document.getElementById('welcome-screen');
const inputEl        = document.getElementById('chat-input');
const sidebar        = document.getElementById('sidebar');
const userLabel      = document.getElementById('current-user-label');
const historyList    = document.getElementById('history-list');
const editOverlay    = document.getElementById('edit-overlay');
const editTextarea   = document.getElementById('edit-textarea');

let currentTab = 'login';

function refreshIcons() {
  if (typeof window.lucide !== 'undefined' && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

/* ══════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════ */
if (token && currentUser) {
  showApp();
  loadHistory();
} else {
  showAuth();
}

/* ── Auto-resize textarea ───────────────────────────── */
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
});

/* Enter = send, Shift+Enter = newline */
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('chat-form').dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  }
});

/* ══════════════════════════════════════════════════════
   AUTH
══════════════════════════════════════════════════════ */
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-login').classList.toggle('active', tab === 'login');
  document.getElementById('tab-register').classList.toggle('active', tab === 'register');
  authBtn.textContent = tab === 'login' ? 'Login' : 'Create Account';
  authError.textContent = '';
}

function showAuth() {
  authOverlay.classList.add('active');
  appContainer.style.display = 'none';
  setTimeout(refreshIcons, 50);
}

function showApp() {
  authOverlay.classList.remove('active');
  appContainer.style.display = 'flex';
  userLabel.textContent = currentUser || '';
  setTimeout(refreshIcons, 50);
}

async function handleAuth(e) {
  e.preventDefault();
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;
  authError.textContent = '';

  try {
    if (currentTab === 'register') {
      const r = await fetch(`${API}/register`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ username, password }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'Registration failed');
    }
    await doLogin(username, password);
  } catch (err) {
    authError.textContent = err.message;
  }
}

async function doLogin(username, password) {
  const body = new URLSearchParams({ username, password });
  const r = await fetch(`${API}/token`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || 'Login failed');

  token       = d.access_token;
  currentUser = username;
  localStorage.setItem(STORE,  token);
  localStorage.setItem(USTORE, username);
  showApp();
  loadHistory();
}

function logout() {
  token       = null;
  currentUser = null;
  localStorage.removeItem(STORE);
  localStorage.removeItem(USTORE);
  resetChat();
  showAuth();
}

/* ══════════════════════════════════════════════════════
   SIDEBAR TOGGLE
══════════════════════════════════════════════════════ */
function toggleSidebar() {
  sidebarOpen = !sidebarOpen;
  sidebar.classList.toggle('collapsed', !sidebarOpen);
}

/* ══════════════════════════════════════════════════════
   CHAT SEND
══════════════════════════════════════════════════════ */
async function sendMessage(e) {
  if (e) e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  hideWelcome();

  appendUserMsg(text);
  const typingEl = appendTyping();
  scrollBottom();

  try {
    const r = await fetch(`${API}/chat`, {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ message: text }),
    });

    if (r.status === 401) { logout(); return; }

    const d = await r.json();
    typingEl.remove();
    appendBotMsg(d.reply);
    scrollBottom();
    loadHistory();
  } catch {
    typingEl.remove();
    appendBotMsg('⚠️ Could not reach the server. Please ensure the server is running.');
    scrollBottom();
  }
}

/* ══════════════════════════════════════════════════════
   MESSAGE BUILDERS
══════════════════════════════════════════════════════ */
function appendUserMsg(text) {
  hasMessages = true;
  const initials = (currentUser || 'U').slice(0, 2).toUpperCase();

  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-body">
      <div class="msg-sender">${escHtml(currentUser || 'You')}</div>
      <div class="msg-bubble user-bubble">${escHtml(text)}</div>
      <div class="msg-actions">
        <button class="action-btn" onclick="copyMsg(this)" title="Copy">
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy
        </button>
        <button class="action-btn" onclick="openEdit(this)" title="Edit & resend">
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          Edit
        </button>
      </div>
    </div>
    <div class="msg-avatar user-av" title="${escHtml(currentUser || 'You')}">${escHtml(initials)}</div>
  `;
  msgsEl.appendChild(row);
  return row;
}

function appendBotMsg(text) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="msg-avatar bot-av" title="AgriBot">🌿</div>
    <div class="msg-body">
      <div class="msg-sender">AgriBot</div>
      <div class="msg-bubble bot-bubble">${formatText(text)}</div>
      <div class="msg-actions">
        <button class="action-btn" onclick="copyMsg(this)" title="Copy">
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy
        </button>
      </div>
    </div>
  `;
  msgsEl.appendChild(row);
  return row;
}

function appendTyping() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'typing-row';
  row.innerHTML = `
    <div class="msg-avatar bot-av">🌿</div>
    <div class="msg-body">
      <div class="msg-sender">AgriBot</div>
      <div class="msg-bubble bot-bubble">
        <div class="typing-bubble">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    </div>
  `;
  msgsEl.appendChild(row);
  return row;
}

/* ══════════════════════════════════════════════════════
   COPY ACTION
══════════════════════════════════════════════════════ */
function copyMsg(btn) {
  const bubble = btn.closest('.msg-body').querySelector('.msg-bubble');
  const text   = bubble.innerText || bubble.textContent;
  navigator.clipboard.writeText(text).then(() => {
    btn.classList.add('copied');
    const orig = btn.innerHTML;
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
    setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('copied'); }, 2000);
  });
}

/* ══════════════════════════════════════════════════════
   EDIT ACTION
══════════════════════════════════════════════════════ */
function openEdit(btn) {
  const bubble   = btn.closest('.msg-body').querySelector('.msg-bubble');
  editingMsgEl   = bubble;
  editTextarea.value = bubble.innerText || bubble.textContent;
  editOverlay.classList.add('open');
  editTextarea.focus();
}

function closeEdit() {
  editOverlay.classList.remove('open');
  editingMsgEl = null;
}

async function submitEdit() {
  const newText = editTextarea.value.trim();
  if (!newText) return;

  // Update the original bubble text
  if (editingMsgEl) {
    editingMsgEl.textContent = newText;
  }
  closeEdit();

  // Remove all messages AFTER the edited one, then resend
  const row = editingMsgEl?.closest('.msg-row');
  if (row) {
    // Remove all following siblings
    let next = row.nextElementSibling;
    while (next) {
      const tmp = next.nextElementSibling;
      next.remove();
      next = tmp;
    }
  }

  // Resend as new query
  inputEl.value = newText;
  await sendMessage(null);
}

/* Click outside edit card to close */
editOverlay.addEventListener('click', e => {
  if (e.target === editOverlay) closeEdit();
});

/* ══════════════════════════════════════════════════════
   SUGGESTION CHIPS
══════════════════════════════════════════════════════ */
function useChip(chip) {
  // Strip the icon span
  const icon = chip.querySelector('.chip-icon');
  const text = chip.innerText.replace(icon ? icon.innerText : '', '').trim();
  inputEl.value = text;
  inputEl.focus();
  sendMessage(null);
}

/* ══════════════════════════════════════════════════════
   HISTORY SIDEBAR
══════════════════════════════════════════════════════ */
async function loadHistory() {
  if (!token) return;
  try {
    const r = await fetch(`${API}/history`, {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (r.status === 401) { logout(); return; }

    const d = await r.json();
    historyList.innerHTML = '';

    const items = [...d.history].reverse().slice(0, 15);
    if (!items.length) {
      historyList.innerHTML = `<div style="padding:12px 14px;color:var(--text-muted);font-size:12px;">No conversations yet</div>`;
      return;
    }

    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span title="${escHtml(item.user_msg)}">${escHtml(truncate(item.user_msg, 32))}</span>
      `;
      historyList.appendChild(div);
    });
  } catch { /* ignore */ }
}

/* ══════════════════════════════════════════════════════
   WELCOME / RESET
══════════════════════════════════════════════════════ */
function hideWelcome() {
  if (welcomeScreen && welcomeScreen.parentNode) {
    welcomeScreen.style.display = 'none';
  }
}

function startNewChat() {
  resetChat();
}

function resetChat() {
  hasMessages = false;
  // Remove all message rows (not the welcome screen)
  const rows = msgsEl.querySelectorAll('.msg-row');
  rows.forEach(r => r.remove());
  if (welcomeScreen) welcomeScreen.style.display = 'flex';
}

/* ══════════════════════════════════════════════════════
   UTILITIES
══════════════════════════════════════════════════════ */
function scrollBottom() {
  msgsEl.scrollTop = msgsEl.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* Light markdown: newlines → <br>, **bold**, bullet • */
function formatText(str) {
  return escHtml(str)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n•/g, '\n<br>•')
    .replace(/\n/g, '<br>');
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + '…' : str;
}
