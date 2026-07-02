function importVideoToProject(filePath, markerTime, isProxy, fullResVideoPath) {
    try {
        if(!isProxy){
            var importSuccess = app.project.importFiles([filePath],true, app.project.rootItem, false);
        }else{
            var importSuccess = app.project.importFiles([fullResVideoPath],true, app.project.rootItem, false);
            var newItem = app.project.rootItem.children[app.project.rootItem.children.numItems - 1];

            var proxyFileExists = new File(filePath).exists;
            var fullResFileExists = new File(fullResVideoPath).exists;
            app.setSDKEventMessage(
                "ClipSeek (debug): proxy exists=" + proxyFileExists + " path=" + filePath +
                " | fullRes exists=" + fullResFileExists + " path=" + fullResVideoPath +
                " | importSuccess=" + importSuccess + " newItem=" + (newItem ? newItem.name : "null"),
                "info"
            );

            // Premiere can still be finishing media analysis on the just-imported
            // clip; attachProxy() fails intermittently if called too soon after
            // importFiles() returns, so retry briefly before giving up.
            var x = false;
            for (var attempt = 0; attempt < 5 && !x; attempt++) {
                if (attempt > 0) { $.sleep(300); }
                x = newItem.attachProxy(filePath, 0);
            }
            if (!x){app.setSDKEventMessage("Proxy failed to attach.", "info");}
        }
        if (!importSuccess) {
            throw new Error('Failed to import video.');
        }

        var newProjectItem = app.project.rootItem.children[app.project.rootItem.children.numItems - 1];

        if (newProjectItem && newProjectItem.type === ProjectItemType.CLIP) {
            var newMarker = newProjectItem.getMarkers().createMarker(parseFloat(markerTime));
            newMarker.name = "Point of interest"; 
        }
    } catch (error) {
        app.setSDKEventMessage(error.message, "error");
    }
}


function selectFolder() {
    var folder = Folder.selectDialog("Select a folder");
    return folder ? folder.fsName : null;
}


function revealInFinder(filePath) {
    var folder = new Folder(filePath);
    if (folder.exists) {
        folder.execute(); // Opens the folder in Finder/Explorer
    }
}

function getSelectedClipFilePath() {
    if (app.project && app.project.rootItem && app.project.rootItem.children.numItems > 0) {
        return app.getCurrentProjectViewSelection()[0].getMediaPath();
    }
    return null;
}

