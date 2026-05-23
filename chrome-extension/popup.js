const conceptInput = document.getElementById('conceptInput');
const searchBtn = document.getElementById('searchBtn');
const loadingEl = document.getElementById('loading');
const resultsEl = document.getElementById('results');
const BACKEND_URL = 'http://localhost:8000/analyze';

searchBtn.addEventListener('click', () => handleSearch(conceptInput.value.trim()));
conceptInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    handleSearch(conceptInput.value.trim());
  }
});

function setLoading(isLoading) {
  loadingEl.style.display = isLoading ? 'block' : 'none';
}

function escapeHtml(text) {
  return text.replace(/[&<>\"]/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;'
  }[char] || char));
}

async function handleSearch(concept) {
  if (!concept) {
    resultsEl.innerHTML = '<div class="message error">Please enter a concept first.</div>';
    return;
  }

  resultsEl.innerHTML = '';
  setLoading(true);

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const tab = tabs[0];
    if (!tab || !tab.url.includes('youtube.com/watch')) {
      setLoading(false);
      resultsEl.innerHTML = '<div class="message error">Open a YouTube watch page first.</div>';
      return;
    }

    try {
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: tab.url, concept }),
      });
      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        resultsEl.innerHTML = `<div class="message error">${escapeHtml(data.detail || 'Backend error occurred.')}</div>`;
        return;
      }

      renderResults(data, concept);
    } catch (error) {
      setLoading(false);
      resultsEl.innerHTML = '<div class="message error">Cannot reach backend at localhost:8000.</div>';
    }
  });
}

function renderResults(data, concept) {
  let html = '';
  html += '<div class="pill-row">';
  html += data.expanded_concepts.map((kw) => `<span class="pill">${escapeHtml(kw)}</span>`).join('');
  html += '</div>';

  if (!data.results || data.results.length === 0) {
    html += `<div class="message">No matches for <strong>"${escapeHtml(concept)}"</strong>.</div>`;
    if (data.suggested_topics && data.suggested_topics.length) {
      html += '<div class="message">Related topics:</div>';
      html += data.suggested_topics.map((topic) => `<span class="suggestion" data-topic="${escapeHtml(topic)}">${escapeHtml(topic)}</span>`).join('');
    }
    resultsEl.innerHTML = html;
    bindSuggestionClicks();
    return;
  }

  resultsEl.innerHTML = html + data.results.map((item) => {
    const highlighted = item.matched_concepts.reduce((text, kw) => {
      if (kw.length < 3) return text;
      return text.replace(new RegExp(`(${kw.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')})`, 'gi'), '<span class="highlight">$1</span>');
    }, escapeHtml(item.text));

    return `
      <div class="result-card">
        <div class="clip-top">
          <span class="clip-range">${escapeHtml(item.formatted_time)}</span>
          <button class="goto-btn" data-time="${item.timestamp}">Go to Clip</button>
        </div>
        <div class="result-text">${highlighted}</div>
      </div>
    `;
  }).join('');

  resultsEl.querySelectorAll('.goto-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const time = Number(btn.dataset.time);
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs[0];
        if (!tab || !tab.id) return;
        sendJumpMessage(tab.id, time);
      });
    });
  });
}

function bindSuggestionClicks() {
  resultsEl.querySelectorAll('.suggestion').forEach((chip) => {
    chip.addEventListener('click', () => handleSearch(chip.dataset.topic));
  });
}

function sendJumpMessage(tabId, time) {
  chrome.tabs.sendMessage(tabId, { action: 'jump', time }, (response) => {
    if (chrome.runtime.lastError) {
      chrome.scripting.executeScript({
        target: { tabId },
        files: ['content.js']
      }, () => {
        chrome.tabs.sendMessage(tabId, { action: 'jump', time });
      });
    }
  });
}
