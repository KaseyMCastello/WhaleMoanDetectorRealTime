
detections = readtable('L:\CalCOFI\Sonobuoy\data\WhaleMoanDetector_predictions\CC201907.csv');

[uniqueRows, ~, idx] = unique(detections(:, {'Deployment',  'Call'}), 'rows');

% Count occurrences
callCounts = accumarray(idx, 1);

% Add call counts to unique combinations
uniqueRows.CallCount = callCounts;

% Find corresponding latitudes and longitudes for each unique combination
uniqueRows.Latitude = zeros(height(uniqueRows), 1);
uniqueRows.Longitude = zeros(height(uniqueRows), 1);

for i = 1:height(uniqueRows)
    % Find the first occurrence of the deployment to get the corresponding latitude and longitude
    deploymentIdx = find(strcmp(detections.Deployment, uniqueRows.Deployment{i}) & strcmp(detections.Call, uniqueRows.Call{i}), 1);
    uniqueRows.Latitude(i) = detections.CalCOFI_Latitude(deploymentIdx);
    uniqueRows.Longitude(i) = detections.CalCOFI_Longitude(deploymentIdx);
end

% Convert Call to categorical
uniqueRows.Call = categorical(uniqueRows.Call);

% Create a table for geobubble
bubbleData = uniqueRows(:, {'Latitude', 'Longitude', 'Call', 'CallCount'});

% Create geobubble plot
figure;
geobubble(bubbleData, 'Latitude', 'Longitude', 'SizeVariable', 'CallCount', 'ColorVariable', 'Call');

% Add title
title('Acoustic Detections by Call Type at CalCOFI Stations');