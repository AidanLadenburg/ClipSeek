'use strict';

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const CLIPSEEK_UI_PREFIX = 'CLIPSEEK_UI ';

/**
 * Resolve search backend: packaged io.exe, or io.py + Python for local testing.
 * - Put io.exe under extension/python/io.exe for production (paths are relative to this file in extension/js/).
 * - Without io.exe, looks for io.py at extension root (parent of js/) and runs CLIPSEEK_PYTHON or `python`.
 */
function resolveSearchBackend(sdkLog) {
  const forceScript = process.env.CLIPSEEK_USE_IO_PY === '1';

  const exeCandidates = [
    path.join(__dirname, 'python', 'io.exe'),
    path.join(__dirname, '..', 'python', 'io.exe'),
  ];

  if (!forceScript) {
    for (const p of exeCandidates) {
      if (fs.existsSync(p)) {
        return { kind: 'exe', command: p, args: [], cwd: path.dirname(p) };
      }
    }
  }

  const ioPyCandidates = [
    path.join(__dirname, '..', 'io.py'),
    path.join(__dirname, 'io.py'),
  ];
  for (const ioPy of ioPyCandidates) {
    if (fs.existsSync(ioPy)) {
      const py = process.env.CLIPSEEK_PYTHON || 'python';
      if (sdkLog) {
        sdkLog('ClipSeek: using io.py for testing (no io.exe). Python: ' + py);
      }
      return {
        kind: 'script',
        command: py,
        args: [ioPy],
        cwd: path.dirname(ioPy),
      };
    }
  }

  return {
    kind: 'exe',
    command: exeCandidates[0],
    args: [],
    cwd: path.dirname(exeCandidates[0]),
  };
}

function parseClipseekUiLine(line, onClipseekUiEvent, getDebugMode, sdkLog) {
  const trimmed = line.trim();
  if (!trimmed.startsWith(CLIPSEEK_UI_PREFIX)) return;
  try {
    const payload = JSON.parse(trimmed.slice(CLIPSEEK_UI_PREFIX.length));
    if (onClipseekUiEvent && payload && typeof payload.message === 'string') {
      onClipseekUiEvent(payload);
    }
  } catch (e) {
    if (getDebugMode && sdkLog) sdkLog('ClipSeek: bad UI line: ' + trimmed);
  }
}

function createSearchBridge({
  sdkLog,
  getDebugMode,
  onClipseekUiEvent,
  onPythonReady,
  onPythonStop,
}) {
  let pythonProcess = null;
  let processReady = false;
  let stdoutBuffer = '';

  function attachJsonStdoutParser(onJsonLine) {
    pythonProcess.stdout.on('data', (data) => {
      stdoutBuffer += data.toString();
      const lines = stdoutBuffer.split('\n');
      stdoutBuffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        if (line.trim().startsWith(CLIPSEEK_UI_PREFIX)) {
          parseClipseekUiLine(line, onClipseekUiEvent, getDebugMode, sdkLog);
          continue;
        }
        try {
          const result = JSON.parse(line);
          if (result.error) sdkLog('Python: ' + result.error);
          /* Search responses may include both error and results[] — always pass through. */
          onJsonLine(result);
        } catch {
          /* non-JSON log lines */
        }
      }
    });
  }

  async function startPythonProcess(onJsonLine) {
    if (pythonProcess) return;

    const embeddingFolder = localStorage.getItem('embeddingFolder') || '';
    const backend = resolveSearchBackend(sdkLog);
    const spawnArgs =
      backend.kind === 'exe'
        ? embeddingFolder
          ? [embeddingFolder]
          : []
        : embeddingFolder
          ? [...backend.args, embeddingFolder]
          : [...backend.args];

    pythonProcess = spawn(backend.command, spawnArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: backend.cwd,
      env: process.env,
    });

    await new Promise((resolve, reject) => {
      let bootBuf = '';
      function onBoot(data) {
        bootBuf += data.toString();
        const parts = bootBuf.split('\n');
        bootBuf = parts.pop() || '';
        for (const line of parts) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (getDebugMode()) sdkLog(trimmed);
          if (trimmed.startsWith(CLIPSEEK_UI_PREFIX)) {
            parseClipseekUiLine(line, onClipseekUiEvent, getDebugMode, sdkLog);
          }
          if (trimmed === 'READY') {
            processReady = true;
            if (onPythonReady) onPythonReady(true);
            pythonProcess.stdout.removeListener('data', onBoot);
            resolve();
            return;
          }
        }
      }
      pythonProcess.stdout.on('data', onBoot);

      pythonProcess.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) sdkLog('Python: ' + msg);
      });
      pythonProcess.on('error', reject);
    });

    const initialEmbeddingFolder = localStorage.getItem('embeddingFolder') || '';
    const initialVideoFolder = localStorage.getItem('videoFolder') || '';

    if (initialEmbeddingFolder) {
      pythonProcess.stdin.write(
        JSON.stringify({
          command: 'update_embedding_folder',
          embedding_folder: initialEmbeddingFolder,
          video_folder: initialVideoFolder,
        }) + '\n'
      );
    }

    attachJsonStdoutParser(onJsonLine);
  }

  function stopPythonProcess() {
    if (pythonProcess) {
      pythonProcess.stdin.write('exit\n');
      pythonProcess.kill();
      pythonProcess = null;
      processReady = false;
      stdoutBuffer = '';
      if (onPythonStop) onPythonStop();
    }
  }

  function sendJson(obj) {
    if (pythonProcess) pythonProcess.stdin.write(JSON.stringify(obj) + '\n');
  }

  return {
    resolveSearchBackend: () => resolveSearchBackend(null),
    startPythonProcess,
    stopPythonProcess,
    sendJson,
    get process() {
      return pythonProcess;
    },
    get ready() {
      return processReady;
    },
  };
}

module.exports = { createSearchBridge, resolveSearchBackend };
