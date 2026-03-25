'use strict';

const path = require('path');

function createResultsController({ csInterface, getFullResPath, sdkLog }) {
  const selectedVideos = new Set();
  let displayedCount = 20;
  let allFiles = [];

  function clearSelectionUI() {
    document.querySelectorAll('.video-item.selected').forEach((item) => item.classList.remove('selected'));
  }

  function updateBulkActionVisibility() {
    const show = selectedVideos.size > 0;
    ['importSelectedBtn', 'searchSimilarBtn', 'clearSelectedBtn'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.hidden = !show;
    });
  }

  function displayResults(files) {
    allFiles = files;
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (!files.length) {
      resultsDiv.innerHTML = '<p class="results-empty">No videos found.</p>';
      return;
    }

    const videosToDisplay = files.slice(0, displayedCount);

    videosToDisplay.forEach((file) => {
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

      const progressBar = document.createElement('div');
      progressBar.className = 'progress-bar';
      videoContainer.appendChild(progressBar);

      const poiMarker = document.createElement('div');
      poiMarker.className = 'point-of-interest';

      videoThumb.addEventListener('loadedmetadata', () => {
        const dur = videoThumb.duration || 1;
        const poiPercent = (time / dur) * 100;
        poiMarker.style.left = `${poiPercent}%`;
      });

      videoContainer.appendChild(poiMarker);

      videoContainer.addEventListener('mousemove', (event) => {
        const rect = videoThumb.getBoundingClientRect();
        const xPos = event.clientX - rect.left;
        const percent = xPos / rect.width;
        const dur = videoThumb.duration || 1;
        videoThumb.currentTime = percent * dur;
        progressBar.style.width = `${percent * 100}%`;
        poiMarker.style.opacity = '1';
      });

      videoContainer.addEventListener('mouseleave', () => {
        videoThumb.pause();
        videoThumb.currentTime = time;
        progressBar.style.width = '0';
        poiMarker.style.opacity = '0';
      });

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
        updateBulkActionVisibility();
      });

      videoContainer.addEventListener('dblclick', () => {
        const proxyPath = document.getElementById('proxyLocation').value.trim();
        const fullResPath = document.getElementById('fullResLocation').value.trim();
        const useProxy = document.getElementById('proxySwitch').checked;

        const normalizedSelectedVideoPath = vid.replace(/\\/g, '/');
        const fullResVideoPath = getFullResPath(
          proxyPath,
          fullResPath,
          normalizedSelectedVideoPath,
          useProxy ? sdkLog : null
        );
        const fullResExists = fullResVideoPath && require('fs').existsSync(fullResVideoPath);

        csInterface.evalScript(
          `importVideoToProject("${normalizedSelectedVideoPath.replace(/\\/g, '\\\\')}", "${time}", ${useProxy && fullResExists},"${fullResVideoPath}")`
        );
      });

      videoContainer.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        const menu = document.getElementById('contextMenu');
        window.__clipseekContext = { videoPath: vid, time };
        menu.style.display = 'block';
        menu.style.left = `${event.pageX}px`;
        menu.style.top = `${event.pageY}px`;
      });

      const titleDiv = document.createElement('div');
      titleDiv.className = 'video-title';
      titleDiv.textContent = path.basename(vid);

      videoContainer.appendChild(videoThumb);
      videoContainer.appendChild(titleDiv);
      resultsDiv.appendChild(videoContainer);
    });

    const loadMoreBtn = document.getElementById('loadMoreBtn');
    loadMoreBtn.hidden = displayedCount >= files.length;
  }

  function importSelectedVideos() {
    const fs = require('fs');
    const proxyPath = document.getElementById('proxyLocation').value.trim();
    const fullResPath = document.getElementById('fullResLocation').value.trim();
    const useProxy = document.getElementById('proxySwitch').checked;

    selectedVideos.forEach((objKey) => {
      const { videoPath, time } = JSON.parse(objKey);
      const normalizedSelectedVideoPath = videoPath.replace(/\\/g, '/');
      const fullResVideoPath = getFullResPath(
        proxyPath,
        fullResPath,
        normalizedSelectedVideoPath,
        useProxy ? sdkLog : null
      );
      const fullResExists = fullResVideoPath && fs.existsSync(fullResVideoPath);
      csInterface.evalScript(
        `importVideoToProject("${normalizedSelectedVideoPath.replace(/\\/g, '\\\\')}", "${time}", ${useProxy && fullResExists},"${fullResVideoPath}")`
      );
    });
    clearSelected();
  }

  function clearSelected() {
    selectedVideos.clear();
    clearSelectionUI();
    updateBulkActionVisibility();
  }

  function searchSimilar(bridge, selectedFolders, queryPath) {
    let query = queryPath;
    if (!query && selectedVideos.size > 0) {
      query = JSON.parse([...selectedVideos][0]).videoPath;
      query = query.replace(/\\\\/g, '\\');
    }
    if (!query) return;

    sdkLog('Similar search: ' + query);

    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;
    const searchRequest = {
      video_folder: selectedFolders.videoFolder,
      embedding_folder: selectedFolders.embeddingFolder,
      annotation_folder: '',
      query,
      is_mean: 'true',
      query_type: 'video',
    };
    if (dateFrom && dateTo) {
      searchRequest.date_from = dateFrom;
      searchRequest.date_to = dateTo;
    }
    bridge.sendJson(searchRequest);
    clearSelected();
  }

  function resetDisplayCount() {
    displayedCount = 20;
  }

  function loadMore() {
    displayedCount += 10;
    displayResults(allFiles);
  }

  return {
    displayResults,
    importSelectedVideos,
    clearSelected,
    searchSimilar,
    resetDisplayCount,
    loadMore,
  };
}

module.exports = { createResultsController };
