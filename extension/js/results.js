'use strict';

const path = require('path');

/**
 * `position:fixed` is viewport-relative; must use clientX/Y (not pageX/Y) or the menu
 * drifts by scroll amount and ends up off-screen for lower rows.
 */
function positionContextMenu(menu, clientX, clientY) {
  const pad = 6;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  menu.style.display = 'block';
  let x = clientX;
  let y = clientY;
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  const r = menu.getBoundingClientRect();
  if (r.right > vw - pad) {
    x -= r.right - (vw - pad);
  }
  if (r.bottom > vh - pad) {
    y -= r.bottom - (vh - pad);
  }
  if (x < pad) x = pad;
  if (y < pad) y = pad;
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
}

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

  function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const metaLine = document.getElementById('searchMetaLine');
    let files = data;
    let meta = null;

    if (Array.isArray(data)) {
      files = data;
    } else if (data && typeof data === 'object') {
      if (Array.isArray(data.results)) {
        files = data.results;
        meta = data;
      } else {
        files = [];
        meta = data.error ? data : null;
      }
    }

    if (metaLine) {
      if (meta && typeof meta.search_seconds === 'number') {
        const parts = [`total ${meta.search_seconds.toFixed(2)}s`];
        if (typeof meta.load_seconds === 'number') {
          parts.push(`cache load ${meta.load_seconds.toFixed(2)}s`);
        }
        if (typeof meta.retrieve_seconds === 'number') {
          parts.push(`similarity ${meta.retrieve_seconds.toFixed(2)}s`);
        }
        let modeLabel = 'Exact GPU batched similarity';
        if (meta.faiss_used) {
          modeLabel = 'FAISS candidate search + exact rerank';
          if (meta.faiss_reason) {
            modeLabel += ` (${meta.faiss_reason})`;
          }
        } else if (meta.faiss_requested && meta.faiss_available === false) {
          modeLabel = 'Exact (install faiss-cpu for FAISS mode)';
        }
        if (meta.faiss_requested && !meta.faiss_used && meta.faiss_available) {
          const reason = meta.faiss_reason ? `: ${meta.faiss_reason}` : '';
          modeLabel = `Exact (FAISS skipped${reason})`;
        }
        if (meta.search_note) {
          modeLabel = meta.search_note;
        }
        metaLine.textContent = `${modeLabel} - ${parts.join(' - ')}`;
        metaLine.hidden = false;
        sdkLog(`ClipSeek: ${modeLabel}. ${parts.join(', ')}.`);
      } else {
        metaLine.hidden = true;
        metaLine.textContent = '';
      }
    }

    allFiles = files;
    resultsDiv.innerHTML = '';

    if (!files.length) {
      resultsDiv.innerHTML =
        '<p class="results-empty">' + (meta && meta.error ? 'Search failed.' : 'No videos found.') + '</p>';
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
        positionContextMenu(menu, event.clientX, event.clientY);
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

  function getSearchMode() {
    const el = document.getElementById('searchModeSwitch');
    if (!el) return 'exact';
    return el.checked ? 'faiss' : 'exact';
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
      search_mode: getSearchMode(),
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
