async function uploadPDF() {
    let file = document.getElementById("pdfFile").files[0];
    let statusDiv = document.getElementById("uploadStatus");
    
    // Check if file selected
    if (!file) {
        statusDiv.innerText = "⚠️ Please select a file first";
        statusDiv.className = "error";
        return;
    }

    // Check file format
    const fileType = file.type;
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (fileType !== 'application/pdf' && fileExtension !== 'pdf') {
        statusDiv.innerText = "❌ Only PDF files are allowed!";
        statusDiv.className = "error";
        return;
    }

    // Check file size < 50 MB
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        statusDiv.innerText = `❌ File too large! Max size: 50MB (Current: ${(file.size / 1024 / 1024).toFixed(2)}MB)`;
        statusDiv.className = "error";
        return;
    }

    // Uploading status
    statusDiv.innerText = "⏳ Uploading... Please wait";
    statusDiv.className = "";

    let formData = new FormData();
    formData.append("file", file);

    try {
        let response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        let result = await response.json();
        console.log(result);
        
        if (response.ok) {
            // ✅ Upload successful message with details
            statusDiv.innerText = `✅ Upload successful! 
📄 Title: "${result.title || 'Untitled'}" 
📊 Chunks: ${result.chunks || 0} 
📁 File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`;
            statusDiv.className = "success";
            
            // Clear file selection
            document.getElementById("pdfFile").value = "";
        } else {
            // Server error
            statusDiv.innerText = `❌ Upload failed: ${result.detail || result.message || 'Unknown error'}`;
            statusDiv.className = "error";
        }
    } catch (error) {
        statusDiv.innerText = "❌ Upload failed! Please check if the server is running";
        statusDiv.className = "error";
        console.error("Upload error:", error);
    }
}

async function ask() {
    let question = document.getElementById("question").value;
    let answerDiv = document.getElementById("answer");
    
    if (!question.trim()) {
        answerDiv.innerText = "⚠️ Please enter a question";
        return;
    }

    // Thinking status
    answerDiv.innerText = "🤔 Thinking...";

    try {
        let response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: question
            })
        });
        
        let result = await response.json();
        
        if (response.ok) {
            answerDiv.innerText = result.answer || "No answer received";
        } else {
            answerDiv.innerText = `❌ Error: ${result.detail || result.message || 'Unknown error'}`;
        }
    } catch (error) {
        answerDiv.innerText = "❌ Request failed! Please check if the server is running";
        console.error("Chat error:", error);
    }
}