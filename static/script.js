// document.addEventListener("DOMContentLoaded", () => {
//     fetch('/data')
//         .then(response => response.json())
//         .then(data => {
//             document.getElementById('users').textContent = data.users;
//             document.getElementById('sales').textContent = `$${data.sales}`;
//             document.getElementById('visitors').textContent = data.visitors;
//         })
//         .catch(error => console.error('Error fetching data:', error));
// });
function updateTime() {
    const timeElement = document.querySelector('.time');
    const now = new Date();
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' };
    timeElement.textContent = now.toLocaleString('en-US', options);
}

// Update the time immediately and every second
updateTime();
setInterval(updateTime, 1000);


document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('videoUploadForm');
    const videoInput = document.getElementById('videoInput');
    const uploadStatus = document.getElementById('uploadStatus');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Check if a file is selected
        if (!videoInput.files.length) {
            uploadStatus.textContent = 'Please select a video file';
            uploadStatus.style.color = 'red';
            return;
        }

        // Validate file type
        const file = videoInput.files[0];
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime'];
        if (!allowedTypes.includes(file.type)) {
            uploadStatus.textContent = 'Invalid file type. Please upload MP4, AVI, or MOV files.';
            uploadStatus.style.color = 'red';
            return;
        }

        // Validate file size (16MB limit)
        const maxSize = 300 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            uploadStatus.textContent = 'File is too large. Maximum size is 16MB.';
            uploadStatus.style.color = 'red';
            return;
        }
        
        const formData = new FormData(form);
        
        try {
            uploadStatus.textContent = 'Uploading...';
            uploadStatus.style.color = 'black';
            
            // Detailed fetch with full error handling
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            // Log full response for debugging
            console.log('Full Response:', response);
            
            // Try to parse JSON even if response is not OK
            const result = await response.json().catch(err => {
                console.error('JSON Parsing Error:', err);
                throw new Error('Could not parse server response');
            });
            
            if (response.ok) {
                uploadStatus.textContent = 'Video uploaded and processed successfully!';
                uploadStatus.style.color = 'green';
                
                // Optional: Reset form
                form.reset();
            } else {
                // Use the error from server response or a generic message
                const errorMessage = result.error || 'Unknown server error';
                uploadStatus.textContent = `Error: ${errorMessage}`;
                uploadStatus.style.color = 'red';
            }
        } catch (error) {
            // Detailed network error logging
            console.error('Full Upload Error:', error);
            
            // More informative error messages
            if (error.name === 'TypeError') {
                uploadStatus.textContent = 'Network error. Please check your internet connection.';
            } else if (error.message.includes('Failed to fetch')) {
                uploadStatus.textContent = 'Could not connect to the server. Please try again.';
            } else {
                uploadStatus.textContent = `Unexpected error: ${error.message}`;
            }
            
            uploadStatus.style.color = 'red';
        }
    });
});