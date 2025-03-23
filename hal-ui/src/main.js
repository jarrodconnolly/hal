import { createRootSpan, fetchSpan, endRootSpan } from './otel.js';
const terminal = document.getElementById('terminal');
let prompt = document.querySelector('.prompt');
let sessionId = null;
let userId = null;
let activeResponse = null;
const API_HOST = 'localhost';
const API_PORT = 8001;
let chunkCount = 0;
let ws = null;

function connectWebSocket() {
  ws = new WebSocket(`ws://${API_HOST}:${API_PORT}/ws/hal`);

  ws.onopen = () => {
    console.log('WebSocket ws.onopen');
  };

  ws.onmessage = (event) => {
    console.log('WebSocket ws.onmessage');
    const message = JSON.parse(event.data);
    const msgType = message.type;
    console.log('Message:', message);
    if (msgType === 'login_response' || msgType === 'logout_response' || msgType === 'query_response') {
      const spanId = message.traceparent?.split('-')[2] ?? null;
      const rootSpan  = fetchSpan(spanId)
      if (rootSpan ) {
        console.log('rootSpan:', rootSpan );

        rootSpan .setAttribute('response', message.error || 'success');
        if (msgType === 'query_response' && !message.done) return;

        console.log('ending root span');
        endRootSpan(spanId);
      } else {
        console.log('root span found');
      }
    }

    if (msgType === 'login_response') {
      if (message.error) {
        displayText(`${message.error}`);
        return;
      }
      sessionId = message.session_id;
      userId = message.user_id;
      displayText(`Logged in as ${userId} with session ${sessionId}`);
    } else if (msgType === 'logout_response') {
      if (message.error) {
        displayText(`${message.error}`);
        return;
      }
      sessionId = null;
      userId = null;
      displayText('Logged out');
    } else if (msgType === 'query_response') {
      handleQueryResponse(message);
    } else if (msgType === 'stats') {
      handleStats(message);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket ws.onclose');
    updateStatus('Disconnected');
    setTimeout(connectWebSocket, 1000);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    updateStatus('Connection error');
  };
}

function login(username, password) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.error('WebSocket not ready');
    return;
  }
  const { traceparent } = createRootSpan('login');

  ws.send(JSON.stringify({ type: 'login', username, password, traceparent }));
}

function logout() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.error('WebSocket not ready');
    return;
  }
  const { traceparent } = createRootSpan('logout');
  ws.send(JSON.stringify({ type: 'logout', session_id: sessionId, traceparent }));
}

function addNewPrompt() {
  const qaPair = document.createElement('div');
  qaPair.className = 'qa-pair';

  const newPrompt = document.createElement('div');
  newPrompt.className = 'prompt activePrompt';
  newPrompt.setAttribute('contenteditable', 'true');
  newPrompt.innerHTML = '<span class="cursor">█</span>';

  qaPair.appendChild(newPrompt);
  terminal.appendChild(qaPair);

  prompt = newPrompt;
  prompt.focus();
  newPrompt.addEventListener('paste', handlePaste);
  terminal.scrollTop = terminal.scrollHeight;
}

function updateStatus(message) {
  const statusElement = document.getElementById('status');
  if (statusElement) {
    statusElement.textContent = message;
  }
}

function handleStats(stats) {
  const { chunk_count = chunkCount, generation = 0, ttfb = 0 } = stats;
  chunkCount = chunk_count;
  updateStatus(
    `Chunks: ${chunkCount} | ` +
    `TTFB: ${ttfb.toFixed(2)}s | ` +
    `Gen: ${generation.toFixed(2)}s | `
  );
}

function handlePaste(e) {
  e.preventDefault();
  navigator.clipboard.readText()
    .then(text => {
      const cursor = prompt.querySelector('.cursor');
      cursor.before(text);
      terminal.scrollTop = terminal.scrollHeight;
    })
    .catch(err => {
      console.error('Failed to paste:', err);
    });
}

function handleQueryResponse(message) {
  const { content, done } = message;

  if (!activeResponse) {
    displayText('');
    // fetch the last response div
    activeResponse = document.querySelectorAll('.response')[document.querySelectorAll('.response').length - 1];
  }

  activeResponse.textContent += content;
  terminal.scrollTop = terminal.scrollHeight;

  if (done) {
    activeResponse = null;
  }
}

function displayText(text) {
  const qaPair = document.createElement('div');
  qaPair.className = 'qa-pair';

  const questionDiv = document.createElement('div');
  questionDiv.className = 'question';
  questionDiv.textContent = prompt.textContent.replace('█', '').trim();
  qaPair.appendChild(questionDiv);

  const responseDiv = document.createElement('div');
  responseDiv.className = 'response';
  //responseDiv.textContent = text;
  responseDiv.innerHTML = text; // clean this
  qaPair.appendChild(responseDiv);

  // Clear the old prompt and remove it
  prompt.removeAttribute('contenteditable');
  prompt.parentElement.removeChild(prompt);

  terminal.appendChild(qaPair);
  addNewPrompt();
}



terminal.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const input = prompt.textContent.replace('█', '').trim();
    if (!input || !ws || ws.readyState !== WebSocket.OPEN) return;

    if (input.startsWith('/')) {
      const [command, ...args] = input.slice(1).split(' ');
      switch (command.toLowerCase()) {
        case 'help':
          displayText('COMMANDS: <br>&nbsp;&nbsp;/login [user] [pass]<br>&nbsp;&nbsp;/logout<br>&nbsp;&nbsp;/mode [green|amber]<br>&nbsp;&nbsp;/help');
          break;
        case 'login':
          const [username, password] = args;
          login(username, password);
          break;
        case 'logout':
          logout();
          break;
        case 'mode':
          const mode = args[0]?.toLowerCase();
          if (mode === 'green') {
            document.body.style.setProperty('--text-color', '#33FF33');
            document.body.style.setProperty('--response-color', '#00CC00');
            document.body.style.setProperty('--cursor-color', '#00FF00');
            document.body.style.setProperty('--bg-color', '#0a1a0a');
            displayText('Mode set to P1 Green');
          } else if (mode === 'amber') {
            document.body.style.setProperty('--text-color', '#FFBF00'); // P3 amber
            document.body.style.setProperty('--response-color', '#E6A800'); // Darker amber
            document.body.style.setProperty('--cursor-color', '#FFD700'); // Bright amber
            document.body.style.setProperty('--bg-color', '#1A1200');
            displayText('Mode set to P3 Amber');
          } else {
            displayText('Usage: /mode [green|amber]');
          }
          break;
        default:
          displayText(`UNKNOWN COMMAND: /${command}`);
      }
    } else {
      const { traceparent } = createRootSpan('query');
      ws.send(JSON.stringify({ type: 'query', session_id: sessionId, query: input, traceparent }));
    }
  } else if (e.key.length === 1 && !e.ctrlKey) {
    e.preventDefault();
    const text = prompt.textContent.replace('█', '');
    prompt.innerHTML = text + e.key + '<span class="cursor">█</span>';
    terminal.scrollTop = terminal.scrollHeight;
  } else if (e.key === 'Backspace') {
    e.preventDefault();
    const text = prompt.textContent.replace('█', '');
    if (text) {
      prompt.innerHTML = text.slice(0, -1) + '<span class="cursor">█</span>';
      terminal.scrollTop = terminal.scrollHeight;
    }
  }
});

document.addEventListener('keydown', (e) => {
  if (document.activeElement === prompt || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key.length > 1 && e.key !== 'Backspace' && e.key !== 'Delete') return;
  prompt.focus();
});


function crtFlicker() {
  const body = document.body;
  setInterval(() => {
    const brightness = 0.9 + Math.random() * 0.1;
    body.style.filter = `brightness(${brightness}) grayscale(0.4)`;
  }, 200);
}

document.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  addNewPrompt();
  crtFlicker();
});
