const csInterface = new CSInterface();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let pythonProcess = null;
let processReady = false;

let debugMode = JSON.parse(localStorage.getItem('debugMode') || "false");
let isProxy = JSON.parse(localStorage.getItem('isProxy') || "false");



// Load saved proxy and full resolution locations
document.getElementById('proxyLocation').value = localStorage.getItem('proxyLocation') || '';
document.getElementById('fullResLocation').value = localStorage.getItem('fullResLocation') || '';

document.getElementById('proxyLocation').addEventListener('change', () => {
    localStorage.setItem('proxyLocation', document.getElementById('proxyLocation').value);
});

document.getElementById('fullResLocation').addEventListener('change', () => {
    localStorage.setItem('fullResLocation', document.getElementById('fullResLocation').value);
});

// Function to start the Python process
async function startPythonProcess() {
    if (pythonProcess) return;


    const pythonPath = path.join(__dirname, './python/io.exe');

    const embeddingFolder = localStorage.getItem('embeddingFolder') || "";
    const args = embeddingFolder ? [embeddingFolder] : [];

    pythonProcess = spawn(pythonPath, args, {
        stdio: ['pipe', 'pipe', 'pipe']
    });

    // Create a promise that resolves when the process is ready
    await new Promise((resolve, reject) => {
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString().trim();
            if(debugMode){print(output);}
            
            if (output === 'READY') {
                print("Model Loaded");
                processReady = true;
                resolve();
            }
            if (output === 'VIDEOS LOADED') {
                print("Videos Loaded");
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            print("ERROR");
            //console.error(`Python Error: ${data}`);
        });

        pythonProcess.on('error', (error) => {
            print("ERROR2");
            reject(error);
        });

    });

    const initialEmbeddingFolder = localStorage.getItem('embeddingFolder') || "";
    const initialVideoFolder = localStorage.getItem('videoFolder') || "";

    if (initialEmbeddingFolder) {
        const initCommand = JSON.stringify({
            command: 'update_embedding_folder',
            embedding_folder: initialEmbeddingFolder,
            video_folder: initialVideoFolder
        });
        pythonProcess.stdin.write(initCommand + '\n');
    }

    // Set up line buffering for stdout
    let buffer = '';
    pythonProcess.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep the last partial line in the buffer
        
        for (const line of lines) {
            if (line.trim()) {
                try {
                    const result = JSON.parse(line);
                    if (result.error) {
                        print("PYTHON ERROR");
                        //alert(result.error);
                        //console.error('Python process error:', result.error);
                    } else {
                        //print("ALL Good");
                        displayResults(result);
                    }
                } catch (e) {
                    //console.error('Error parsing Python output:', e);
                }
            }
        }
    });
}

// Function to stop the Python process
function stopPythonProcess() {
    if (pythonProcess) {
        pythonProcess.stdin.write('exit\n');
        pythonProcess.kill();
        pythonProcess = null;
        processReady = false;
    }
}

// Start process when extension loads
startPythonProcess().catch(console.error);

// Clean up when extension closes
window.onbeforeunload = () => {
    stopPythonProcess();
};

const settingsBtn = document.getElementById('settingsBtn');
const mainPage = document.getElementById('mainPage');
const settingsPage = document.getElementById('settingsPage');
const backBtn = document.getElementById('backBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');

settingsBtn.addEventListener('click', () => {
    mainPage.style.display = 'none';
    settingsPage.style.display = 'block';
});

backBtn.addEventListener('click', () => {
    mainPage.style.display = 'block';
    settingsPage.style.display = 'none';
});

document.getElementById('annotationPageBtn').addEventListener('click', () => {
    document.getElementById('settingsPage').style.display = 'none';
    document.getElementById('annotationPage').style.display = 'block';
    document.getElementById('settingsBtn').style.display = 'none'; // Hide gear icon
});

// Handle folder selection for annotation
document.getElementById('annotationFolderSelectBtn').addEventListener('click', () => {
    csInterface.evalScript(`selectFolder()`, (folderPath) => {
        if (folderPath && folderPath != 'null') {
            document.getElementById('annotationFolder').value = folderPath;
            localStorage.setItem('annotationFolder', folderPath); // Save persistently
        }
    });

});

document.getElementById('backToSettingsBtn').addEventListener('click', () => {
    localStorage.setItem('annotationFolder', document.getElementById('annotationFolder').value);
    selectedFolders.annotationFolder = document.getElementById('annotationFolder').value;
    document.getElementById('annotationPage').style.display = 'none'; 
    document.getElementById('settingsPage').style.display = 'block';
    document.getElementById('settingsBtn').style.display = 'block'; // Show gear icon
});

const annotationMedium = document.getElementById('annotationMedium');
annotationMedium.addEventListener('change', () => {
    document.getElementById('annotationImageContainer').style.display = annotationMedium.value === 'image' ? 'block' : 'none';
    document.getElementById('annotationTextContainer').style.display = annotationMedium.value === 'text' ? 'block' : 'none';
    document.getElementById('annotationVideoContainer').style.display = annotationMedium.value === 'video' ? 'block' : 'none';
});

const imageInput = document.getElementById('annotationImage');
imageInput.setAttribute('multiple', true);

document.getElementById('saveAnnotationBtn').addEventListener('click', () => {
    const annotationKey = document.getElementById('annotationKey').value;
    const annotationMediumValue = document.getElementById('annotationMedium').value;
    const annotationFolder = document.getElementById('annotationFolder').value;
    let annotationValues = [];
    if (annotationKey != "" && annotationKey != null){
        if (annotationMediumValue === 'text') {
            annotationValues.push(document.getElementById('annotationText').value);
        } else if (annotationMediumValue === 'image') {
            for (const file of imageInput.files) {
                annotationValues.push(file.path);
            }
        } else if (annotationMediumValue === 'video') {
            annotationValues.push(document.getElementById('annotationVideo').value);
        }

        annotationValues.forEach((value) => {
            const annotationCommand = JSON.stringify({
                command: 'create_annotation',
                annotation_folder: annotationFolder,
                key: annotationKey,
                type: annotationMediumValue,
                value: value
            });
            pythonProcess.stdin.write(annotationCommand + '\n');
        });
    }
    localStorage.setItem('annotationFolder', document.getElementById('annotationFolder').value);
    selectedFolders.annotationFolder = document.getElementById('annotationFolder').value;
    // Return to settings page
    document.getElementById('annotationPage').style.display = 'none';
    document.getElementById('settingsPage').style.display = 'block';
    document.getElementById('settingsBtn').style.display = 'block';
});

saveSettingsBtn.addEventListener('click', () => {
    const newEmbeddingFolder = document.getElementById('embeddingFolderInput').value;
    const currentEmbeddingFolder = selectedFolders.embeddingFolder;

    const newVideoFolder = document.getElementById('videoFolderInput').value;
    const currentVideoFolder = selectedFolders.embeddingFolder;

    const proxyLocation = document.getElementById('proxyLocation').value;
    localStorage.setItem('proxyLocation', proxyLocation);
    const fullResLocation = document.getElementById('fullResLocation').value;
    localStorage.setItem('fullResLocation', fullResLocation);

    debugMode = document.getElementById('debugSwitch').checked;
    localStorage.setItem('debugMode', JSON.stringify(debugMode));

    isProxy = document.getElementById('proxySwitch').checked;
    localStorage.setItem('isProxy', JSON.stringify(isProxy));

    if (newEmbeddingFolder !== currentEmbeddingFolder) {
        selectedFolders.embeddingFolder = newEmbeddingFolder;
        localStorage.setItem('embeddingFolder', newEmbeddingFolder);
    }

    if (newVideoFolder !== currentVideoFolder) {
        selectedFolders.videoFolder = newVideoFolder;
        localStorage.setItem('videoFolder', newVideoFolder);
    }

    mainPage.style.display = 'block';
    settingsPage.style.display = 'none';
});


document.getElementById('searchButton').addEventListener('click', async () => {
    const searchInput = document.getElementById('search');
    const event = { key: 'Enter', target: searchInput }; // Simulate Enter key event
    await handleSearchKeyPress(event);
});

const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');

// Trigger the file input dialog
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

// Handle file selection
fileInput.addEventListener('change', (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
        const filePath = selectedFile.path; // Get the file path
        const fileType = selectedFile.type.split('/')[0]; // Determine file type
        handleFileSearch(filePath, fileType);
    }
});

// Function to handle file-based search
function handleFileSearch(filePath, fileType) {
    const queryType = fileType === 'image' ? 'image' : fileType === 'video' ? 'video' : null;
    if (!queryType) {
        //alert('Unsupported file type. Please upload an image or video.');
        return;
    }

    const searchRequest = {
        command: 'search_file',
        video_folder: selectedFolders.videoFolder,
        embedding_folder: selectedFolders.embeddingFolder,
        annotation_folder: selectedFolders.annotationFolder,
        file_path: filePath,
        query_type: queryType,
    };

    pythonProcess.stdin.write(JSON.stringify(searchRequest) + '\n');
}

document.getElementById('meanMaxSwitch').addEventListener('change', (event) => {
    localStorage.setItem('isMean', JSON.stringify(event.target.checked));
});

document.getElementById('debugSwitch').addEventListener('change', (event) => {
    localStorage.setItem('debugMode', JSON.stringify(event.target.checked));
});

const proxySwitch = document.getElementById('proxySwitch');
const proxySettings = document.getElementById('proxySettings');
const proxyLocationInput = document.getElementById('proxyLocation');
const fullResLocationInput = document.getElementById('fullResLocation');

if (isProxy) {
    proxySettings.style.display = 'block';
} else {
    proxySettings.style.display = 'none';
}

proxySwitch.addEventListener('change', () => {
    if (proxySwitch.checked) {
        proxySettings.style.display = 'block';
    } else {
        proxySettings.style.display = 'none';
        proxyLocationInput.value = '';
        fullResLocationInput.value = '';
    }
});

document.getElementById('proxyLocationSelectBtn').addEventListener('click', () => {
    csInterface.evalScript(`selectFolder()`, (folderPath) => {
        if (folderPath && folderPath !== 'null') {
            proxyLocationInput.value = folderPath;
        }
    });
});

document.getElementById('fullResLocationSelectBtn').addEventListener('click', () => {
    csInterface.evalScript(`selectFolder()`, (folderPath) => {
        if (folderPath && folderPath !== 'null') {
            fullResLocationInput.value = folderPath;
        }
    });
});

const contextMenu = document.getElementById('contextMenu');
const enlargeOption = document.getElementById('enlargeOption');
const importOption = document.getElementById('importOption');
const searchSimilarOption = document.getElementById('searchSimilarOption');

const videoModal = document.getElementById('videoModal');
const modalVideo = document.getElementById('modalVideo');
const closeModal = document.getElementById('closeModal');

let selectedVideoPath = '';
let selectedVideoTime = 0;

document.addEventListener('click', () => {
    contextMenu.style.display = 'none'; // Hide menu on outside click
});

// Function to show the modal with a slight delay to avoid flickering
function showModal() {
    videoModal.classList.add('visible');
    setTimeout(() => {
        modalVideo.src = selectedVideoPath; // Set source after modal is visible
        modalVideo.currentTime = selectedVideoTime;
    }, 10); // Short delay to allow for any layout settling
}

// Function to hide the modal and clear the video source
function hideModal() {
    modalVideo.src = ''; // Clear the source to stop playback
    videoModal.classList.remove('visible');
}

// Event listener for the enlarge option in the context menu
enlargeOption.addEventListener('click', () => {
    if (selectedVideoPath) {
        showModal(); // Show modal and load video
        contextMenu.style.display = 'none';
    }
});

// Close the modal when the "X" button is clicked
closeModal.addEventListener('click', hideModal);


// Close the modal when clicking outside the modal content
window.addEventListener('click', (event) => {
    if (event.target === videoModal) {
        hideModal();
    }
});

importOption.addEventListener('click', () => {
    if (selectedVideoPath) {
        print("Importing video: "+ selectedVideoPath);
        proxyPath = document.getElementById('proxyLocation').value.trim();
        fullResPath = document.getElementById('fullResLocation').value.trim();
        isProxy = document.getElementById('proxySwitch').checked;
        normalizedSelectedVideoPath = selectedVideoPath.replace(/\\/g, '/');
        fullResVideoPath = getFullResPath(proxyPath, fullResPath, normalizedSelectedVideoPath);
            
        fullResExists = fullResVideoPath && fs.existsSync(fullResVideoPath);
        
        csInterface.evalScript(`importVideoToProject("${normalizedSelectedVideoPath.replace(/\\/g, '\\\\')}", "${selectedVideoTime}", ${isProxy && fullResExists},"${fullResVideoPath}")`);
        contextMenu.style.display = 'none';
    }
});

searchSimilarOption.addEventListener('click', () => {
    if (selectedVideoPath) {
        searchSimilar(selectedVideoPath); // Pass the specific video path
        contextMenu.style.display = 'none';
    }
});

const revealInFinderOption = document.getElementById('revealInFinderOption');

revealInFinderOption.addEventListener('click', () => {
    if (selectedVideoPath) {
        const filePath = selectedVideoPath;
        const parentFolder = path.dirname(filePath).replace(/\\/g, '/');
        csInterface.evalScript(`revealInFinder("${parentFolder}")`);
        contextMenu.style.display = 'none';
    }
});


const filterButton = document.getElementById('filterButton');
const filterMenu = document.getElementById('filterMenu');

filterButton.addEventListener('click', () => {
    const filterMenu = document.getElementById('filterMenu');
    const isHidden = filterMenu.style.display === 'none';
    
    if (isHidden) {
        filterMenu.style.display = 'flex'; // Set to flexbox layout when shown
        filterMenu.style.gap = '10px';
        filterMenu.style.alignItems = 'center';
    } else {
        filterMenu.style.display = 'none'; // Hide the menu
    }
});


function getFullResPath(proxyPath, fullResPath, videoPath) {
    normalizedProxyPath = proxyPath.replace(/\\/g, '/');
    normalizedFullResPath = fullResPath.replace(/\\/g, '/');
    normalizedVideoPath = videoPath.replace(/\\/g, '/');
    if (!normalizedVideoPath.startsWith(normalizedProxyPath)) {
        return null;
    }
    var replaced = normalizedVideoPath.replace(normalizedProxyPath, normalizedFullResPath);
    if (fs.existsSync(replaced)) {return replaced;}
   
    if (fs.existsSync(replaced.replace('.mov.mp4','.mov'))) {return replaced.replace('.mov.mp4','.mov');}
    if (fs.existsSync(replaced.replace('.MXF.mp4','.MXF'))) {return replaced.replace('.MXF.mp4','.MXF');}
  
    if (fs.existsSync(replaced.replace('.mp4.mp4','.mp4'))) {return replaced.replace('.mp4.mp4','.mp4');}
    if (fs.existsSync(replaced.replace('.mp3.mp4','.mp3'))) {return replaced.replace('.mp3.mp4','.mp3');}
    if (fs.existsSync(replaced.replace('.wav.mp4','.wav'))) {return replaced.replace('.wav.mp4','.wav');}
    if (fs.existsSync(replaced.replace('.avi.mp4','.avi'))) {return replaced.replace('.avi.mp4','.avi');}
    if (fs.existsSync(replaced.replace('.mkv.mp4','.mkv'))) {return replaced.replace('.mkv.mp4','.mkv');}
    if (fs.existsSync(replaced.replace('.webm.mp4','.webm'))) {return replaced.replace('.webm.mp4','.webm');}
    if (fs.existsSync(replaced.replace('.hevc.mp4','.hevc'))) {return replaced.replace('.hevc.mp4','.hevc');}
    //R3D files
    p = replaced.split("/");
    g = replaced.split("/")[p.length-1];
    g = g.slice(0,g.length-4);
    p[p.length-1] = g+".RDC";
    p.push(g+"_001.R3D");
    p = p.join("/");
    
    return p;
}


// Load saved values on page load
document.getElementById('videoFolderInput').value = localStorage.getItem('videoFolder') || "";
document.getElementById('embeddingFolderInput').value = localStorage.getItem('embeddingFolder') || "";
document.getElementById('search').addEventListener('keypress', handleSearchKeyPress, false);
document.getElementById('meanMaxSwitch').checked = JSON.parse(localStorage.getItem('isMean') || "true");
document.getElementById('debugSwitch').checked = JSON.parse(localStorage.getItem('debugMode') || "false");
document.getElementById('proxySwitch').checked = JSON.parse(localStorage.getItem('isProxy') || "false");
document.getElementById('proxyLocation').value = localStorage.getItem('proxyLocation') || '';
document.getElementById('fullResLocation').value = localStorage.getItem('fullResLocation') || '';
document.getElementById('annotationFolder').value = localStorage.getItem('annotationFolder') || '';

document.getElementById('videoFolderSelectBtn').addEventListener('click', () => openFolderDialog('video'));
document.getElementById('embeddingFolderSelectBtn').addEventListener('click', () => openFolderDialog('embedding'));

let selectedFolders = {
    videoFolder: localStorage.getItem('videoFolder') || "",
    embeddingFolder: localStorage.getItem('embeddingFolder') || "",
    annotationFolder: localStorage.getItem('annotationFolder') || ""
};

let isOpen = false;
async function openFolderDialog(type) {
    if (isOpen) {return}
    
    isOpen = true;
    csInterface.evalScript(`selectFolder()`, (folderPath) => {
        isOpen = false;
        if (folderPath && folderPath!='null') {
            if (type === 'video') {
                document.getElementById('videoFolderInput').value = folderPath;
                localStorage.setItem('videoFolder', folderPath);
            } else {
                document.getElementById('embeddingFolderInput').value = folderPath;
                localStorage.setItem('embeddingFolder', folderPath);
            }
        }
    });
    csInterface.evalScript("app.bringToFront();"); // Force Premiere Pro to bring dialog to front
}

function print(value) {
    const strin = 'app.setSDKEventMessage("' + value + '", "info")';
    csInterface.evalScript(strin);
}

async function handleSearchKeyPress(event) {
    if (event.key === 'Enter' && event.target.value) {
        //print("START");
        displayedCount = 20;
        selectedVideos.clear();
        const query = event.target.value;
        const isMean = JSON.parse(localStorage.getItem('isMean') || "true");
        const dateFrom = document.getElementById('dateFrom').value;
        const dateTo = document.getElementById('dateTo').value;
        // Ensure process is running
        if (!pythonProcess || !processReady) {
            try {
                await startPythonProcess();
            } catch (error) {
                print("Error starting search process");
                return;
            }
        }
        try {
            const searchRequest = {
                video_folder: selectedFolders.videoFolder,
                embedding_folder: selectedFolders.embeddingFolder,
                annotation_folder: selectedFolders.annotationFolder,
                query: query,
                is_mean: isMean,
                query_type: "text",
            };

            if (dateFrom && dateTo) {
                searchRequest.date_from = dateFrom;
                searchRequest.date_to = dateTo;
            }
            
            // Send the search request to the Python process
            pythonProcess.stdin.write(JSON.stringify(searchRequest) + '\n');
        } catch (error) {
            print("Search error");
        }
    }
}


document.getElementById('importSelectedBtn').addEventListener('click', importSelectedVideos);
document.getElementById('searchSimilarBtn').addEventListener('click', () => {searchSimilar();});
document.getElementById('clearSelectedBtn').addEventListener('click', clearSelected);

function importSelectedVideos() {
    selectedVideos.forEach(objKey => {  
        const { videoPath, time } = JSON.parse(objKey); // Parse JSON string to object
        //selectedVideoPath = videoPath;
        const proxyPath = document.getElementById('proxyLocation').value.trim();
        const fullResPath = document.getElementById('fullResLocation').value.trim();
        const isProxy = document.getElementById('proxySwitch').checked;

        const normalizedSelectedVideoPath = videoPath.replace(/\\/g, '/');
        fullResVideoPath = getFullResPath(proxyPath, fullResPath, normalizedSelectedVideoPath);
            
            
        fullResExists = fullResVideoPath && fs.existsSync(fullResVideoPath);
        
        csInterface.evalScript(`importVideoToProject("${normalizedSelectedVideoPath.replace(/\\/g, '\\\\')}", "${time}", ${isProxy && fullResExists},"${fullResVideoPath}")`);
    });
    clearSelected();
}

function clearSelected() {
    selectedVideos.clear();
    document.querySelectorAll('.video-item.selected').forEach(item => item.classList.remove('selected'));
}

function searchSimilar(videoPath) {
    let searchVids = [];
    let query = ""
    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;
    if (videoPath) {
        query = videoPath;
    } else if (selectedVideos.size > 0) {
        query = JSON.parse([...selectedVideos][0]).videoPath;
        query = query.replace(/\\\\/g, '\\');
    }
    print("sim search to: " + query);
    try {
        const searchRequest = {
            video_folder: selectedFolders.videoFolder,
            embedding_folder: selectedFolders.embeddingFolder,
            annotation_folder: "",
            query: query,
            is_mean: "true",
            query_type: "video"
        };
        if (dateFrom && dateTo) {
            searchRequest.date_from = dateFrom;
            searchRequest.date_to = dateTo;
        }
        pythonProcess.stdin.write(JSON.stringify(searchRequest) + '\n');
    } catch (error) {
        print("Search error");
    }

    clearSelected();
}

let selectedVideos = new Set(); // Set to store selected videos
let displayedCount = 20; // Initial number of displayed videos
let allFiles = []; // Variable to store all search results

function displayResults(files) {
    allFiles = files; // Store all files for future "Load More" operations

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (files.length === 0) {
        resultsDiv.innerHTML = '<p>No videos found.</p>';
        return;
    }

    // Display only up to the current count
    const videosToDisplay = files.slice(0, displayedCount);

    videosToDisplay.forEach(file => {
        const time = file[1];
        const vid = file[0];

        const videoContainer = document.createElement('div');
        videoContainer.className = 'video-item';

        const videoThumb = document.createElement('video');
        videoThumb.className = 'video-thumb';
        videoThumb.loading = 'lazy';
        videoThumb.src = vid;
        videoThumb.currentTime = time;
        videoThumb.volume = 0.1;

        // Create progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        videoContainer.appendChild(progressBar);

        // Create point of interest marker
        const poiMarker = document.createElement('div');
        poiMarker.className = 'point-of-interest';

        // Position the marker based on point of interest time
        videoThumb.addEventListener('loadedmetadata', () => {
            const poiPercent = (time / videoThumb.duration) * 100;
            poiMarker.style.left = `${poiPercent}%`;
        });

        videoContainer.appendChild(poiMarker);

        // scrub through video
        videoContainer.addEventListener('mousemove', (event) => {
            const rect = videoThumb.getBoundingClientRect();
            const xPos = event.clientX - rect.left;
            const percent = xPos / rect.width;
            videoThumb.currentTime = percent * videoThumb.duration;

            // Update progress bar width
            progressBar.style.width = `${percent * 100}%`;
            poiMarker.style.opacity = 1; // Show the marker
        });

        // Pause video on mouse leave
        videoContainer.addEventListener('mouseleave', () => {
            videoThumb.pause();
            videoThumb.currentTime = time;
            progressBar.style.width = '0';
            poiMarker.style.opacity = 0; // Hide the marker
        });

        // Click to select
        videoContainer.addEventListener('click', () => {
            const videoPath = vid.replace(/\\/g, '\\\\');
            const obj = { videoPath, time };
            const objKey = JSON.stringify(obj);

            if (selectedVideos.has(objKey)) {
                selectedVideos.delete(objKey);
                videoContainer.classList.remove('selected');
            } else {
                selectedVideos.add(objKey);
                videoContainer.classList.add('selected');
            }
            const importButton = document.getElementById('importSelectedBtn');
            const searchSimilar = document.getElementById('searchSimilarBtn');
            const clearSelected = document.getElementById('clearSelectedBtn');
            if (selectedVideos.size > 0) {
                importButton.style.display = 'block';
                searchSimilar.style.display = 'block';
                clearSelected.style.display = 'block';
            } else {
                importButton.style.display = 'none';
                searchSimilar.style.display = 'none';
                clearSelected.style.display = 'none';
            }
        });

        // Double-click to import immediately

        videoContainer.addEventListener('dblclick', () => {
            const proxyPath = document.getElementById('proxyLocation').value.trim();
            const fullResPath = document.getElementById('fullResLocation').value.trim();
            const isProxy = document.getElementById('proxySwitch').checked;

            const normalizedSelectedVideoPath = vid.replace(/\\/g, '/');
            fullResVideoPath = getFullResPath(proxyPath, fullResPath, normalizedSelectedVideoPath);
            
            
            fullResExists = fullResVideoPath && fs.existsSync(fullResVideoPath);
            //print(fullResVideoPath);
            
            csInterface.evalScript(`importVideoToProject("${normalizedSelectedVideoPath.replace(/\\/g, '\\\\')}", "${time}", ${isProxy && fullResExists},"${fullResVideoPath}")`);
            
        });

        videoContainer.addEventListener('contextmenu', (event) => {
            event.preventDefault();
            selectedVideoPath = vid;
            selectedVideoTime = time;
        
            contextMenu.style.display = 'block';
            contextMenu.style.left = `${event.pageX}px`;
            contextMenu.style.top = `${event.pageY}px`;
        });

        // Title
        const titleDiv = document.createElement('div');
        titleDiv.className = 'video-title';
        titleDiv.textContent = path.basename(vid);

        videoContainer.appendChild(videoThumb);
        videoContainer.appendChild(titleDiv);
        resultsDiv.appendChild(videoContainer);
    });

    // Show or hide the "Load More" button based on the total results
    const loadMoreBtn = document.getElementById('loadMoreBtn');
    loadMoreBtn.style.display = displayedCount < files.length ? 'block' : 'none';
}

// Event listener for "Load More" button
document.getElementById('loadMoreBtn').addEventListener('click', () => {
    displayedCount += 10; // Increment displayed video count by 10
    displayResults(allFiles); // Refresh display with updated count
});