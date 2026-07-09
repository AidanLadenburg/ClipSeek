'use strict';

/**
 * Orchestrates full-res resolution: tries eMAM first when enabled, otherwise
 * (or on any failure) falls back to the existing local path-matching logic.
 */
function createFullResResolver({ getFullResPath, extractEmamUuid, sdkLog }) {
  function isEmamEnabled() {
    try {
      return JSON.parse(localStorage.getItem('emamEnabled') || 'false');
    } catch {
      return false;
    }
  }

  /**
   * Resolve a (possibly proxy) video path to its full-resolution counterpart.
   * Never rejects — always resolves to a path string or null.
   */
  async function resolveFullResPath({ videoPath, proxyPath, fullResPaths, useProxy, bridge, logIfProxy }) {
    if (useProxy && isEmamEnabled() && bridge && bridge.ready) {
      const uuid = extractEmamUuid(videoPath);
      if (uuid) {
        try {
          const result = await bridge.sendJsonRequest({ command: 'resolve_emam_uuid', uuid });
          if (result && result.path) {
            if (logIfProxy) logIfProxy('EMAM resolved full-res path: ' + result.path);
            return result.path;
          }
          if (logIfProxy) {
            logIfProxy('EMAM lookup failed, falling back to local matching: ' + (result && result.error));
          }
        } catch (e) {
          if (logIfProxy) logIfProxy('EMAM request error, falling back to local matching: ' + e.message);
        }
      } else if (logIfProxy) {
        logIfProxy('No EMAM UUID found in proxy filename, falling back to local matching.');
      }
    }
    return getFullResPath(proxyPath, fullResPaths, videoPath, logIfProxy);
  }

  return { resolveFullResPath, isEmamEnabled };
}

module.exports = { createFullResResolver };
