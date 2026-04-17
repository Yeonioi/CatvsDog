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

let currentImage = null;

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
            confidenceLabel.textContent = `Confidence: ${data.confidence}%`;
            confidenceBar.style.width = `${data.confidence}%`;
            messageLabel.textContent = data.message;
            resultFrame.style.display = 'block';
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
