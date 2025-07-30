document.addEventListener('DOMContentLoaded', () => {
    // Correctly target the theme toggle button by its ID
    const themeToggle = document.getElementById('theme-toggle');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const uploadButton = document.getElementById('uploadButton');
    const uploadStatus = document.getElementById('uploadStatus');
    const dataPreview = document.getElementById('dataPreview');
    const previewHeader = document.getElementById('previewHeader');
    const previewBody = document.getElementById('previewBody');
    const functionGrid = document.getElementById('functionGrid');

    // --- Dark/Light Mode Toggle Logic ---
    // Function to set the theme class on the <html> element
    function setTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
            themeToggle.textContent = '☀️'; // Sun icon for light mode
        } else {
            document.documentElement.classList.remove('dark');
            themeToggle.textContent = '🌓'; // Moon icon for dark mode
        }
        localStorage.setItem('theme', theme); // Save preference
    }

    // Check for saved theme preference or system preference on page load
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // If no saved theme, check OS preference
        setTheme('dark');
    } else {
        // Default to light if no saved theme and OS preference is not dark
        setTheme('light');
    }

    // Add event listener to the toggle button
    themeToggle.addEventListener('click', () => {
        // Toggle the 'dark' class on the <html> element
        if (document.documentElement.classList.contains('dark')) {
            setTheme('light');
        } else {
            setTheme('dark');
        }
    });

    // --- File Upload and Data Preview Logic (Existing Code) ---
    // Display selected file name and enable upload button
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = `Selected file: ${fileInput.files[0].name}`;
            uploadButton.disabled = false;
            uploadStatus.textContent = ''; // Clear previous status
            dataPreview.classList.add('hidden'); // Hide preview on new file selection
        } else {
            fileNameDisplay.textContent = '';
            uploadButton.disabled = true;
        }
    });

    // Handle file upload
    window.uploadFile = async () => {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first!';
            uploadStatus.className = 'mt-4 text-sm font-semibold text-red-500';
            return;
        }

        uploadButton.disabled = true;
        uploadButton.textContent = 'Uploading...';
        uploadStatus.textContent = 'Uploading file, please wait...';
        uploadStatus.className = 'mt-4 text-sm font-semibold text-blue-500';

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();

            if (response.ok) {
                uploadStatus.textContent = data.message + ' Data preview below.';
                uploadStatus.className = 'mt-4 text-sm font-semibold text-green-500';

                // Display data preview
                if (data.preview) {
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = data.preview;
                    const table = tempDiv.querySelector('table');

                    if (table) {
                        // Clear existing preview
                        previewHeader.innerHTML = '';
                        previewBody.innerHTML = '';

                        // Copy header
                        const headerRow = table.querySelector('thead tr');
                        if (headerRow) {
                            previewHeader.innerHTML = headerRow.innerHTML;
                        }

                        // Copy body
                        const bodyContent = table.querySelector('tbody');
                        if (bodyContent) {
                            previewBody.innerHTML = bodyContent.innerHTML;
                        }
                        dataPreview.classList.remove('hidden'); // Show preview
                    }
                }

            } else {
                uploadStatus.textContent = `Error: ${data.error}`;
                uploadStatus.className = 'mt-4 text-sm font-semibold text-red-500';
            }
        } catch (error) {
            console.error("Upload failed:", error);
            uploadStatus.textContent = `Upload failed: ${error.message}`;
            uploadStatus.className = 'mt-4 text-sm font-semibold text-red-500';
        } finally {
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload Data';
        }
    };

    // Function to navigate to the operation page
    window.runFunction = (funcName) => {
        window.location.href = `/function/${funcName}`;
    };

    // Dynamically generate function blocks
    // Ensure functionsList is defined before this runs.
    // In your HTML, the <script> passing functionsList should be before main.js.
    if (functionGrid && typeof functionsList !== 'undefined' && functionsList) {
        functionsList.forEach(func => {
            const div = document.createElement("div");
            // Ensure these Tailwind classes are configured for dark mode if desired
            div.className = "p-6 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 shadow-md text-white hover:scale-105 transition-all duration-300 cursor-pointer animate-pulse-custom";
            div.onclick = () => runFunction(func.func_name);

            const h2 = document.createElement("h2");
            h2.className = "text-xl font-semibold mb-2";
            h2.innerText = func.name; // Use the more readable name

            const p = document.createElement("p");
            p.className = "text-sm mt-2 opacity-90";
            p.innerText = `Perform ${func.name.toLowerCase()} operation on dataset.`;

            div.appendChild(h2);
            div.appendChild(p);
            functionGrid.appendChild(div);
        });
    }
});