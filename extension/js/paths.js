'use strict';

const fs = require('fs');

/**
 * Map a proxy path to a full-resolution path (comma-separated candidates).
 */
function getFullResPath(proxyPath, fullResPaths, videoPath, logIfProxy) {
  const normalizedProxyPath = proxyPath.replace(/\\/g, '/');
  const normalizedVideoPath = videoPath.replace(/\\/g, '/');

  if (!normalizedVideoPath.startsWith(normalizedProxyPath)) {
    if (logIfProxy) logIfProxy('Video path does not start with the proxy path.');
    return null;
  }

  const fullResPathList = fullResPaths.split(',').map((s) => s.trim()).filter(Boolean);

  for (let i = 0; i < fullResPathList.length; i++) {
    const candidate = fullResPathList[i];
    const normalizedFullResCandidate = candidate.replace(/\\/g, '/');
    const replaced = normalizedVideoPath.replace(normalizedProxyPath, normalizedFullResCandidate);
    if (logIfProxy) logIfProxy('Trying full res path: ' + replaced);

    if (fs.existsSync(replaced)) {
      if (logIfProxy) logIfProxy('Found full res file at: ' + replaced);
      return replaced;
    }

    const variations = [
      { find: '.mov.mp4', replace: '.mov' },
      { find: '.MXF.mp4', replace: '.MXF' },
      { find: '.mp4.mp4', replace: '.mp4' },
      { find: '.mp3.mp4', replace: '.mp3' },
      { find: '.wav.mp4', replace: '.wav' },
      { find: '.avi.mp4', replace: '.avi' },
      { find: '.mkv.mp4', replace: '.mkv' },
      { find: '.webm.mp4', replace: '.webm' },
      { find: '.hevc.mp4', replace: '.hevc' },
    ];

    for (let j = 0; j < variations.length; j++) {
      const variant = replaced.replace(variations[j].find, variations[j].replace);
      if (fs.existsSync(variant)) {
        if (logIfProxy) {
          logIfProxy('Found full res file with variation (' + variations[j].replace + '): ' + variant);
        }
        return variant;
      }
    }

    const parts = replaced.split('/');
    const lastPart = parts[parts.length - 1];
    const baseName = lastPart.slice(0, lastPart.length - 4);
    parts[parts.length - 1] = baseName + '.RDC';
    parts.push(baseName + '_001.R3D');
    const r3dPath = parts.join('/');
    if (fs.existsSync(r3dPath)) {
      if (logIfProxy) logIfProxy('Found full res R3D file: ' + r3dPath);
      return r3dPath;
    }
  }

  if (logIfProxy) logIfProxy('No matching full resolution file found for video: ' + videoPath);
  return null;
}

module.exports = { getFullResPath };
