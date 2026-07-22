// ===== Utility Functions =====
function getCurrentPage() {
    const path = window.location.pathname;
    return path.split('/').pop() || 'welcome';
}

// ===== Mobile Nav Toggle =====
document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (toggle) {
        toggle.addEventListener('click', function() {
            navLinks.classList.toggle('open');
        });
    }
    
    // Highlight active nav link
    const currentPage = getCurrentPage();
    document.querySelectorAll('.nav-links a').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPage || 
            (currentPage === '' && href === 'welcome') ||
            (href === 'chat' && (currentPage === 'chat' || currentPage === 'symptoms' || currentPage === 'results' || currentPage === 'recommendations'))) {
            link.classList.add('active');
        }
    });
});

// ===== Quick Reply Buttons (chat page) =====
function addQuickReply(text) {
    const input = document.getElementById('chat-input');
    if (input) {
        input.value = text;
        sendChatMessage();
    }
}

// ===== Symptom Picker Logic =====
let selectedSymptoms = new Set();

function toggleSymptom(element, symptom) {
    if (selectedSymptoms.has(symptom)) {
        selectedSymptoms.delete(symptom);
        element.classList.remove('selected');
    } else {
        selectedSymptoms.add(symptom);
        element.classList.add('selected');
    }
    updateSelectedBar();
}

function updateSelectedBar() {
    const bar = document.getElementById('selected-symptoms');
    if (!bar) return;
    
    if (selectedSymptoms.size === 0) {
        bar.innerHTML = '<span style="color: var(--text-muted); font-size: 0.9rem;">Click symptoms above to select them...</span>';
        return;
    }
    
    bar.innerHTML = '';
    selectedSymptoms.forEach(symptom => {
        const tag = document.createElement('span');
        tag.className = 'symptom-tag';
        tag.innerHTML = `${symptom} <span class="remove-symptom" onclick="removeSymptom('${symptom}')">&times;</span>`;
        bar.appendChild(tag);
    });
}

function removeSymptom(symptom) {
    selectedSymptoms.delete(symptom);
    // Also deselect the tag in the grid
    document.querySelectorAll('.symptom-tag').forEach(el => {
        if (el.textContent.trim() === symptom) {
            el.classList.remove('selected');
        }
    });
    updateSelectedBar();
}

function analyzeSelectedSymptoms() {
    if (selectedSymptoms.size === 0) {
        alert('Please select at least one symptom.');
        return;
    }
    
    const symptomsText = Array.from(selectedSymptoms).join(', ');
    // Store in sessionStorage to pass to chat page
    sessionStorage.setItem('symptom_input', symptomsText);
    window.location.href = '/chat';
}

// ===== Chat Functionality =====
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');
    const typing = document.getElementById('typing-indicator');
    const text = input.value.trim();
    
    if (!text) return;
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.innerHTML = `${escapeHtml(text)}<div class="message-time">just now</div>`;
    messages.appendChild(userMsg);
    
    input.value = '';
    messages.scrollTop = messages.scrollHeight;
    
    // Show typing indicator
    typing.classList.add('active');
    messages.scrollTop = messages.scrollHeight;
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        
        typing.classList.remove('active');
        
        // Build bot response
        let responseHtml = `<div class="chat-message bot">`;
        responseHtml += `<p>${escapeHtml(data.response)}</p>`;
        
        // Show top predictions
        if (data.top_predictions && data.top_predictions.length > 0) {
            responseHtml += `<div style="margin-top: 12px;">`;
            data.top_predictions.forEach((p, idx) => {
                const color = idx === 0 ? 'var(--primary)' : 'var(--text-light)';
                const bg = idx === 0 ? 'rgba(37,99,235,0.08)' : 'transparent';
                responseHtml += `
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:0.9rem;">
                        <span style="font-weight:${idx === 0 ? '700' : '500'};color:${color}">${escapeHtml(p.disease)}</span>
                        <span style="color:${color}">${p.confidence}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:${Math.min(p.confidence, 100)}%"></div>
                    </div>
                `;
            });
            responseHtml += `</div>`;
            
            // Add view details button
            responseHtml += `<div style="margin-top: 10px;">`;
            responseHtml += `<a href="/results?symptoms=${encodeURIComponent(text)}" class="btn btn-sm btn-primary">View Full Results →</a>`;
            responseHtml += `</div>`;
        }
        
        responseHtml += `<div class="message-time">just now</div>`;
        responseHtml += `</div>`;
        
        messages.insertAdjacentHTML('beforeend', responseHtml);
        messages.scrollTop = messages.scrollHeight;
        
    } catch (err) {
        typing.classList.remove('active');
        const errMsg = document.createElement('div');
        errMsg.className = 'chat-message bot';
        errMsg.innerHTML = `<p>Sorry, I couldn't process your request. Please try again.</p><div class="message-time">just now</div>`;
        messages.appendChild(errMsg);
        messages.scrollTop = messages.scrollHeight;
    }
}

// Handle chat form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('chat-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            sendChatMessage();
        });
    }
    
    // Handle Enter key in chat input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
    
    // Check for stored symptom input from symptom picker
    const storedInput = sessionStorage.getItem('symptom_input');
    if (storedInput && chatInput) {
        chatInput.value = storedInput;
        sessionStorage.removeItem('symptom_input');
        setTimeout(() => sendChatMessage(), 300);
    }
});

// ===== Emergency Detection =====
document.addEventListener('DOMContentLoaded', function() {
    const banner = document.getElementById('emergency-banner');
    if (banner && banner.classList.contains('active')) {
        // Auto-hide after 10 seconds if not interacted with
        setTimeout(() => {
            // Keep it visible but don't auto-hide emergency
        }, 10000);
    }
});

// ===== Helpers =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

