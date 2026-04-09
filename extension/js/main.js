'use strict';

/**
 * CEP: If Node is off, `require` throws immediately and no click handlers run (dead UI).
 * We show a visible error and defer wiring until DOMContentLoaded.
 */
(function clipseekBootstrap() {
  function showBootError(err) {
    const text =
      (err && err.stack) ||
      (err && err.message) ||
      String(err);
    const lines = [
      'ClipSeek panel failed to start:',
      '',
      text,
      '',
      'If you see "require is not defined" or similar:',
      '• In CSXS/manifest.xml, CEFCommandLine must include --enable-nodejs and --mixed-context',
      '• Fully quit and restart Premiere after changing the manifest',
      '• Copy the whole extension folder (index.html, js/, lib/, jsx/, css/, CSXS/)',
      '',
      'If you see a module / "Cannot find module" error: copy js/paths.js, js/bridge.js, and js/results.js into the extension; main.js loads them via the extension path (require("./…") from CEP often breaks).',
    ].join('\n');

    function paint() {
      const root = document.body || document.documentElement;
      if (!root) return;
      if (document.getElementById('clipseek_boot_error')) return;
      const pre = document.createElement('pre');
      pre.id = 'clipseek_boot_error';
      pre.style.cssText = [
        'margin:12px',
        'padding:12px',
        'background:#2d1518',
        'color:#fecaca',
        'font:12px/1.4 Consolas,monospace',
        'white-space:pre-wrap',
        'word-break:break-word',
        'z-index:2147483647',
        'position:relative',
        'border-radius:8px',
        'border:1px solid #991b1b',
      ].join(';');
      pre.textContent = lines;
      root.insertBefore(pre, root.firstChild);
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', paint);
    } else {
      paint();
    }
  }

  let path;
  let fs;
  let csInterface;
  let getFullResPath;
  let createSearchBridge;
  let createResultsController;

  try {
    if (typeof require !== 'function') {
      throw new Error(
        'require() is not available (CEP Node integration is off for this panel).'
      );
    }
    path = require('path');
    fs = require('fs');

    if (typeof CSInterface === 'undefined') {
      throw new Error('CSInterface is undefined — check that lib/CSInterface.js loads before js/main.js.');
    }
    csInterface = new CSInterface();

    /**
     * CEP: `require('./foo')` from a script-tagged main file often resolves from the host
     * process cwd (not the js/ folder), so local modules fail to load and wireUi never runs.
     * Load sibling modules by absolute path under the extension root from the CEP API.
     */
    function loadJsModule(fileName) {
      const candidates = [];
      try {
        const extRoot = csInterface.getSystemPath(SystemPath.EXTENSION);
        if (extRoot) {
          candidates.push(path.join(extRoot, 'js', fileName));
        }
      } catch {
        /* non-CEP or API unavailable */
      }
      if (typeof __dirname !== 'undefined') {
        candidates.push(path.join(__dirname, fileName));
      }
      for (let i = 0; i < candidates.length; i++) {
        const p = candidates[i];
        if (p && fs.existsSync(p)) {
          return require(p);
        }
      }
      throw new Error(
        `Could not load "${fileName}". Tried:\n${candidates.filter(Boolean).join('\n') || '(none)'}\n\nCopy the full extension folder including js/paths.js, js/bridge.js, and js/results.js.`
      );
    }

    ({ getFullResPath } = loadJsModule('paths.js'));
    ({ createSearchBridge } = loadJsModule('bridge.js'));
    ({ createResultsController } = loadJsModule('results.js'));
  } catch (e) {
    showBootError(e);
    return;
  }

  function sdkLog(message) {
    try {
      const escaped = String(message).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
      csInterface.evalScript(`app.setSDKEventMessage("${escaped}", "info")`);
    } catch {
      /* ignore */
    }
  }

  function getDebugMode() {
    try {
      return JSON.parse(localStorage.getItem('debugMode') || 'false');
    } catch {
      return false;
    }
  }

  function setClipseekReadyIndicator(state) {
    const dot = document.getElementById('clipseekReadyDot');
    if (!dot) return;
    dot.classList.remove('ready-dot--pending', 'ready-dot--ready');
    if (state === 'ready') {
      dot.classList.add('ready-dot--ready');
      dot.title = 'ClipSeek: model and cache ready — you can search.';
    } else {
      dot.classList.add('ready-dot--pending');
      dot.title =
        state === 'stopped'
          ? 'ClipSeek: search backend stopped.'
          : 'ClipSeek: loading model and embedding cache…';
    }
  }

  const bridge = createSearchBridge({
    sdkLog,
    getDebugMode,
    onClipseekUiEvent: (payload) => {
      if (payload && typeof payload.message === 'string') {
        sdkLog(payload.message);
      }
    },
    onPythonReady: () => setClipseekReadyIndicator('ready'),
    onPythonStop: () => setClipseekReadyIndicator('stopped'),
  });
  const results = createResultsController({ csInterface, getFullResPath, sdkLog });

  let selectedFolders = {
    videoFolder: localStorage.getItem('videoFolder') || '',
    embeddingFolder: localStorage.getItem('embeddingFolder') || '',
    annotationFolder: localStorage.getItem('annotationFolder') || '',
  };

  async function ensureBridgeReady() {
    if (!bridge.process || !bridge.ready) {
      await bridge.startPythonProcess((json) => results.displayResults(json));
    }
  }

  function wireUi() {
    function getSearchMode() {
      const el = document.getElementById('searchModeSlider');
      if (!el) return 'exact';
      return el.value === '1' ? 'faiss' : 'exact';
    }

    const mainPage = document.getElementById('mainPage');
    const settingsPage = document.getElementById('settingsPage');
    const annotationPage = document.getElementById('annotationPage');
    if (!mainPage || !settingsPage || !annotationPage) {
      throw new Error('Missing #mainPage, #settingsPage, or #annotationPage — index.html mismatch?');
    }

    setClipseekReadyIndicator('pending');

    ensureBridgeReady().catch((err) => {
      console.error(err);
      sdkLog('Failed to start search backend');
    });

    window.addEventListener('beforeunload', () => bridge.stopPythonProcess());

    document.getElementById('settingsBtn').addEventListener('click', () => {
      mainPage.hidden = true;
      settingsPage.hidden = false;
    });

    document.getElementById('backBtn').addEventListener('click', () => {
      settingsPage.hidden = true;
      mainPage.hidden = false;
    });

    document.getElementById('annotationPageBtn').addEventListener('click', () => {
      settingsPage.hidden = true;
      annotationPage.hidden = false;
      document.getElementById('settingsBtn').style.visibility = 'hidden';
    });

    document.getElementById('backToSettingsBtn').addEventListener('click', () => {
      localStorage.setItem('annotationFolder', document.getElementById('annotationFolder').value);
      selectedFolders.annotationFolder = document.getElementById('annotationFolder').value;
      annotationPage.hidden = true;
      settingsPage.hidden = false;
      document.getElementById('settingsBtn').style.visibility = 'visible';
    });

    document.getElementById('saveSettingsBtn').addEventListener('click', () => {
      const newEmbeddingFolder = document.getElementById('embeddingFolderInput').value;
      const newVideoFolder = document.getElementById('videoFolderInput').value;
      const prevEmbeddingFolder = selectedFolders.embeddingFolder;

      localStorage.setItem('proxyLocation', document.getElementById('proxyLocation').value);
      localStorage.setItem('fullResLocation', document.getElementById('fullResLocation').value);
      localStorage.setItem('debugMode', JSON.stringify(document.getElementById('debugSwitch').checked));
      localStorage.setItem('isProxy', JSON.stringify(document.getElementById('proxySwitch').checked));

      selectedFolders.embeddingFolder = newEmbeddingFolder;
      selectedFolders.videoFolder = newVideoFolder;
      localStorage.setItem('embeddingFolder', newEmbeddingFolder);
      localStorage.setItem('videoFolder', newVideoFolder);

      if (newEmbeddingFolder !== prevEmbeddingFolder) {
        ensureBridgeReady().then(() => {
          bridge.sendJson({
            command: 'update_embedding_folder',
            embedding_folder: newEmbeddingFolder,
            video_folder: newVideoFolder,
          });
        });
      }

      settingsPage.hidden = true;
      mainPage.hidden = false;
    });

    let folderDialogOpen = false;
    function openFolderDialog(type) {
      if (folderDialogOpen) return;
      folderDialogOpen = true;
      csInterface.evalScript(`selectFolder()`, (folderPath) => {
        folderDialogOpen = false;
        if (folderPath && folderPath !== 'null') {
          if (type === 'video') {
            document.getElementById('videoFolderInput').value = folderPath;
            localStorage.setItem('videoFolder', folderPath);
          } else {
            document.getElementById('embeddingFolderInput').value = folderPath;
            localStorage.setItem('embeddingFolder', folderPath);
          }
        }
      });
      csInterface.evalScript('app.bringToFront();');
    }

    document.getElementById('videoFolderSelectBtn').addEventListener('click', () => openFolderDialog('video'));
    document.getElementById('embeddingFolderSelectBtn').addEventListener('click', () => openFolderDialog('embedding'));

    async function handleSearchKeyPress(event) {
      if (event.key !== 'Enter' || !event.target.value.trim()) return;

      results.resetDisplayCount();
      results.clearSelected();

      try {
        await ensureBridgeReady();
      } catch {
        sdkLog('Error starting search process');
        return;
      }

      const isMean = JSON.parse(localStorage.getItem('isMean') || 'true');
      const dateFrom = document.getElementById('dateFrom').value;
      const dateTo = document.getElementById('dateTo').value;

      const searchRequest = {
        video_folder: selectedFolders.videoFolder,
        embedding_folder: selectedFolders.embeddingFolder,
        annotation_folder: selectedFolders.annotationFolder,
        query: event.target.value,
        is_mean: isMean,
        query_type: 'text',
        search_mode: getSearchMode(),
      };
      if (dateFrom && dateTo) {
        searchRequest.date_from = dateFrom;
        searchRequest.date_to = dateTo;
      }
      bridge.sendJson(searchRequest);
    }

    document.getElementById('search').addEventListener('keypress', handleSearchKeyPress);
    document.getElementById('searchButton').addEventListener('click', async () => {
      const input = document.getElementById('search');
      await handleSearchKeyPress({ key: 'Enter', target: input });
    });

    document.getElementById('uploadButton').addEventListener('click', () => {
      document.getElementById('fileInput').click();
    });

    document.getElementById('fileInput').addEventListener('change', (event) => {
      const f = event.target.files[0];
      if (!f) return;
      const fileType = f.type.split('/')[0];
      const queryType = fileType === 'image' ? 'image' : fileType === 'video' ? 'video' : null;
      if (!queryType) return;
      ensureBridgeReady().then(() => {
        bridge.sendJson({
          command: 'search_file',
          video_folder: selectedFolders.videoFolder,
          embedding_folder: selectedFolders.embeddingFolder,
          annotation_folder: selectedFolders.annotationFolder,
          file_path: f.path,
          query_type: queryType,
          is_mean: JSON.parse(localStorage.getItem('isMean') || 'true'),
          search_mode: getSearchMode(),
        });
      });
    });

    document.getElementById('meanMaxSwitch').addEventListener('change', (e) => {
      localStorage.setItem('isMean', JSON.stringify(e.target.checked));
    });

    document.getElementById('debugSwitch').addEventListener('change', (e) => {
      localStorage.setItem('debugMode', JSON.stringify(e.target.checked));
    });

    const proxySwitch = document.getElementById('proxySwitch');
    const proxySettings = document.getElementById('proxySettings');

    function refreshProxySection() {
      proxySettings.hidden = !proxySwitch.checked;
    }

    proxySwitch.addEventListener('change', () => {
      refreshProxySection();
      if (!proxySwitch.checked) {
        document.getElementById('proxyLocation').value = '';
        document.getElementById('fullResLocation').value = '';
      }
    });

    ['proxyLocation', 'fullResLocation'].forEach((id) => {
      document.getElementById(id).addEventListener('change', (e) => {
        localStorage.setItem(id, e.target.value);
      });
    });

    document.getElementById('proxyLocationSelectBtn').addEventListener('click', () => {
      csInterface.evalScript(`selectFolder()`, (folderPath) => {
        if (folderPath && folderPath !== 'null') {
          document.getElementById('proxyLocation').value = folderPath;
        }
      });
    });

    document.getElementById('fullResLocationSelectBtn').addEventListener('click', () => {
      csInterface.evalScript(`selectFolder()`, (folderPath) => {
        if (folderPath && folderPath !== 'null') {
          document.getElementById('fullResLocation').value = folderPath;
        }
      });
    });

    document.getElementById('filterButton').addEventListener('click', () => {
      const menu = document.getElementById('filterMenu');
      menu.hidden = !menu.hidden;
    });

    document.getElementById('importSelectedBtn').addEventListener('click', () => results.importSelectedVideos());
    document.getElementById('searchSimilarBtn').addEventListener('click', () =>
      results.searchSimilar(bridge, selectedFolders, null)
    );
    document.getElementById('clearSelectedBtn').addEventListener('click', () => results.clearSelected());
    document.getElementById('loadMoreBtn').addEventListener('click', () => results.loadMore());

    const contextMenu = document.getElementById('contextMenu');
    const videoModal = document.getElementById('videoModal');
    const modalVideo = document.getElementById('modalVideo');

    document.addEventListener('click', () => {
      contextMenu.style.display = 'none';
    });

    document.getElementById('enlargeOption').addEventListener('click', () => {
      const ctx = window.__clipseekContext;
      if (!ctx) return;
      videoModal.classList.add('visible');
      setTimeout(() => {
        modalVideo.src = ctx.videoPath;
        modalVideo.currentTime = ctx.time;
      }, 10);
      contextMenu.style.display = 'none';
    });

    document.getElementById('closeModal').addEventListener('click', () => {
      modalVideo.src = '';
      videoModal.classList.remove('visible');
    });

    window.addEventListener('click', (event) => {
      if (event.target === videoModal) {
        modalVideo.src = '';
        videoModal.classList.remove('visible');
      }
    });

    document.getElementById('importOption').addEventListener('click', () => {
      const ctx = window.__clipseekContext;
      if (!ctx) return;
      const proxyPath = document.getElementById('proxyLocation').value.trim();
      const fullResPath = document.getElementById('fullResLocation').value.trim();
      const useProxy = document.getElementById('proxySwitch').checked;
      const normalized = ctx.videoPath.replace(/\\/g, '/');
      const fullResVideoPath = getFullResPath(proxyPath, fullResPath, normalized, useProxy ? sdkLog : null);
      const fullResExists = fullResVideoPath && fs.existsSync(fullResVideoPath);
      csInterface.evalScript(
        `importVideoToProject("${normalized.replace(/\\/g, '\\\\')}", "${ctx.time}", ${useProxy && fullResExists},"${fullResVideoPath}")`
      );
      contextMenu.style.display = 'none';
    });

    document.getElementById('searchSimilarOption').addEventListener('click', () => {
      const ctx = window.__clipseekContext;
      if (!ctx) return;
      results.searchSimilar(bridge, selectedFolders, ctx.videoPath);
      contextMenu.style.display = 'none';
    });

    document.getElementById('revealInFinderOption').addEventListener('click', () => {
      const ctx = window.__clipseekContext;
      if (!ctx) return;
      const parentFolder = path.dirname(ctx.videoPath).replace(/\\/g, '/');
      csInterface.evalScript(`revealInFinder("${parentFolder}")`);
      contextMenu.style.display = 'none';
    });

    document.getElementById('annotationFolderSelectBtn').addEventListener('click', () => {
      csInterface.evalScript(`selectFolder()`, (folderPath) => {
        if (folderPath && folderPath !== 'null') {
          document.getElementById('annotationFolder').value = folderPath;
          localStorage.setItem('annotationFolder', folderPath);
        }
      });
    });

    document.getElementById('annotationMedium').addEventListener('change', (e) => {
      const v = e.target.value;
      document.getElementById('annotationImageContainer').hidden = v !== 'image';
      document.getElementById('annotationTextContainer').hidden = v !== 'text';
      document.getElementById('annotationVideoContainer').hidden = v !== 'video';
    });

    const imageInput = document.getElementById('annotationImage');
    imageInput.setAttribute('multiple', 'true');

    document.getElementById('saveAnnotationBtn').addEventListener('click', () => {
      const annotationKey = document.getElementById('annotationKey').value;
      const medium = document.getElementById('annotationMedium').value;
      const annotationFolder = document.getElementById('annotationFolder').value;
      const values = [];
      if (annotationKey) {
        if (medium === 'text') values.push(document.getElementById('annotationText').value);
        else if (medium === 'image') {
          for (const file of imageInput.files) values.push(file.path);
        } else if (medium === 'video') values.push(document.getElementById('annotationVideo').value);
        values.forEach((value) => {
          bridge.sendJson({
            command: 'create_annotation',
            annotation_folder: annotationFolder,
            key: annotationKey,
            type: medium,
            value,
          });
        });
      }
      localStorage.setItem('annotationFolder', annotationFolder);
      selectedFolders.annotationFolder = annotationFolder;
      annotationPage.hidden = true;
      settingsPage.hidden = false;
      document.getElementById('settingsBtn').style.visibility = 'visible';
    });

    const dropZone = document.getElementById('app');
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    dropZone.addEventListener('drop', (event) => {
      event.preventDefault();
      event.stopPropagation();
      const files = event.dataTransfer.files;
      if (files && files.length > 0 && files[0].path) {
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          const fileType = file.type.split('/')[0];
          const queryType = fileType === 'image' ? 'image' : fileType === 'video' ? 'video' : null;
          if (!queryType) continue;
          ensureBridgeReady().then(() => {
            bridge.sendJson({
              command: 'search_file',
              video_folder: selectedFolders.videoFolder,
              embedding_folder: selectedFolders.embeddingFolder,
              annotation_folder: selectedFolders.annotationFolder,
              file_path: file.path,
              query_type: queryType,
              is_mean: JSON.parse(localStorage.getItem('isMean') || 'true'),
              search_mode: getSearchMode(),
            });
          });
        }
      } else {
        csInterface.evalScript('getSelectedClipFilePath()', (result) => {
          if (result && result !== 'null') {
            sdkLog(result);
            ensureBridgeReady().then(() => {
              bridge.sendJson({
                command: 'search_file',
                video_folder: selectedFolders.videoFolder,
                embedding_folder: selectedFolders.embeddingFolder,
                annotation_folder: selectedFolders.annotationFolder,
                file_path: result,
                query_type: 'video',
                is_mean: JSON.parse(localStorage.getItem('isMean') || 'true'),
                search_mode: getSearchMode(),
              });
            });
          } else sdkLog('No valid file path from clip.');
        });
      }
    });

    document.getElementById('videoFolderInput').value = selectedFolders.videoFolder;
    document.getElementById('embeddingFolderInput').value = selectedFolders.embeddingFolder;
    document.getElementById('annotationFolder').value = selectedFolders.annotationFolder;
    document.getElementById('proxyLocation').value = localStorage.getItem('proxyLocation') || '';
    document.getElementById('fullResLocation').value = localStorage.getItem('fullResLocation') || '';
    document.getElementById('meanMaxSwitch').checked = JSON.parse(localStorage.getItem('isMean') || 'true');
    const searchModeSlider = document.getElementById('searchModeSlider');
    if (searchModeSlider) {
      searchModeSlider.value = localStorage.getItem('searchMode') === 'faiss' ? '1' : '0';
      searchModeSlider.addEventListener('input', () => {
        localStorage.setItem('searchMode', searchModeSlider.value === '1' ? 'faiss' : 'exact');
      });
    }
    document.getElementById('debugSwitch').checked = getDebugMode();
    document.getElementById('proxySwitch').checked = JSON.parse(localStorage.getItem('isProxy') || 'false');
    refreshProxySection();
  }

  try {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        try {
          wireUi();
        } catch (e) {
          showBootError(e);
        }
      });
    } else {
      wireUi();
    }
  } catch (e) {
    showBootError(e);
  }
})();
