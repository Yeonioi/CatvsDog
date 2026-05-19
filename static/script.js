// Get DOM elements
const uploadBtn = document.getElementById('uploadBtn');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const fileInput = document.getElementById('fileInput');
const imageLabel = document.getElementById('imageLabel');
const previewImage = document.getElementById('previewImage');
const resultFrame = document.getElementById('resultFrame');
const resultLabel = document.getElementById('resultLabel');
const confidenceLabel = document.getElementById('confidenceLabel');
const confidenceBar = document.getElementById('confidenceBar');
const messageLabel = document.getElementById('messageLabel');
const logsContainer = document.getElementById('logsContainer');
const toggleLogsBtn = document.getElementById('toggleLogsBtn');

let currentImage = null;
let logsVisible = true;

// Fetch and display logs
async function updateLogs() {
    try {
        const response = await fetch('/logs');
        const data = await response.json();
        
        if (data.logs && data.logs.length > 0) {
            let html = '';
            data.logs.forEach(log => {
                let className = 'logs-entry';
                if (log.includes('ERROR')) {
                    className += ' logs-entry-error';
                } else if (log.includes('WARNING')) {
                    className += ' logs-entry-warning';
                } else if (log.includes('INFO')) {
                    className += ' logs-entry-info';
                }
                html += `<div class="${className}">${escapeHtml(log)}</div>`;
            });
            logsContainer.innerHTML = html;
            // Auto-scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        } else {
            logsContainer.innerHTML = '<div class="logs-loading">No logs yet</div>';
        }
    } catch (error) {
        logsContainer.innerHTML = `<div class="logs-loading">Error loading logs: ${error.message}</div>`;
    }
}

// Escape HTML to prevent injection
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toggle logs visibility
toggleLogsBtn.addEventListener('click', () => {
    logsVisible = !logsVisible;
    logsContainer.style.display = logsVisible ? 'block' : 'none';
    toggleLogsBtn.textContent = logsVisible ? 'Hide' : 'Show';
});

// Show conditions modal on page load
window.addEventListener('load', () => {
    const conditionsModal = document.getElementById('conditionsModal');
    const closeBtn = document.querySelector('.close-btn');
    const closeModalBtn = document.getElementById('closeModalBtn');

    // Show modal
    conditionsModal.style.display = 'flex';

    // Close modal functions
    const closeModal = () => {
        conditionsModal.style.display = 'none';
    };

    closeBtn.addEventListener('click', closeModal);
    closeModalBtn.addEventListener('click', closeModal);

    // Close modal when clicking outside of it
    window.addEventListener('click', (event) => {
        if (event.target === conditionsModal) {
            closeModal();
        }
    });

    // Load logs initially and set auto-refresh
    updateLogs();
    setInterval(updateLogs, 2000); // Refresh logs every 2 seconds
});

// Upload button click
uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Display preview
    const reader = new FileReader();
    reader.onload = (event) => {
        currentImage = file;
        previewImage.src = event.target.result;
        previewImage.style.display = 'block';
        imageLabel.style.display = 'none';
        predictBtn.disabled = false;
        
        // Clear previous results
        resultFrame.style.display = 'none';
    };
    reader.readAsDataURL(file);
});

// Predict button click
predictBtn.addEventListener('click', async () => {
    if (!currentImage) {
        alert('Please upload an image first!');
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    try {
        const formData = new FormData();
        formData.append('file', currentImage);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Display results
            resultLabel.textContent = data.label;
            resultLabel.style.color = data.color;
            
            // Only show confidence if it's not null
            if (data.confidence !== null) {
                confidenceLabel.textContent = `Confidence: ${data.confidence}%`;
                confidenceLabel.style.display = 'block';
                confidenceBar.style.width = `${data.confidence}%`;
                confidenceBar.style.display = 'block';
            } else {
                confidenceLabel.style.display = 'none';
                confidenceBar.style.display = 'none';
                confidenceBar.style.width = '0%';
            }
            
            messageLabel.textContent = data.message;
            resultFrame.style.display = 'block';
            
            // Update logs after prediction
            updateLogs();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Failed to predict: ' + error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
});

// Clear button click
clearBtn.addEventListener('click', () => {
    currentImage = null;
    fileInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    imageLabel.style.display = 'block';
    imageLabel.textContent = '📸 No image loaded\n\nClick \'Upload Image\' to select a file';
    resultFrame.style.display = 'none';
    predictBtn.disabled = true;
    confidenceBar.style.width = '0%';
});
