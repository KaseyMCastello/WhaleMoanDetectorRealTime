

% Sonobuoy effort and spatial maps....
% read in acoustic detections and acoustic effort 

detections = readtable('L:\WhaleMoanDetector\labeled_data\logs\CalCOFI\CC201907.xls');
effort = readtable('L:\Acoustic_Effort_CalCOFI_merged\1907BH\1907BH_Acoustic_Effort.xlsx');
CalCOFI_stations = readtable('L:\CalCOFI\Sonobuoy\data\CalCOFIStationOrder.csv');
outDir ='L:\CalCOFI\Sonobuoy\data\WhaleMoanDetector_predictions';


% parse detections by station...

% Initialize arrays to store results
matchingDeployments = {};
matchingSBLatitudes = [];
matchingSBLongitudes = [];
matchingTransect_ID = {};
matchingStation_ID = {};
matchingDetections = {};  
matchingInputFile = {};
matchingSpeciesCodes = {};
matchingCalls = {};
matchingStartTimes = [];  % Initialize as empty datetime array
matchingEndTimes = [];    % Initialize as empty datetime array
matchingCalCOFILatitudes = [];
matchingCalCOFILongitudes = [];
% Specify the columns you want to keep from the detections table

% Regular expression to match the deployment name (e.g., 'SB03')
pattern = 'SB\d+';
% Find matching rows
for i = 1:height(detections)
    inputfile = detections.InputFile{i};
    tokens = regexp(inputfile, pattern, 'match');    
    if ~isempty(tokens)
        deploymentPart = tokens{1}; % The matched deployment name
        
        % Find matching deployment in the first spreadsheet
        matchIdx = find(strcmp(effort.Deployment_Name, deploymentPart));
        
        if ~isempty(matchIdx)
            % Store the matching information
            matchingDeployments{end+1} = deploymentPart; 
            matchingSBLatitudes(end+1) = effort.Latitude(matchIdx);
            matchingSBLongitudes(end+1) = effort.Longitude(matchIdx);
            matchingTransect_ID{end+1} = effort.Transect_ID(matchIdx);
            matchingStation_ID{end+1} = effort.Station_ID(matchIdx);
            
             % Append the relevant columns from the detections table
            matchingInputFile{end+1} = detections.InputFile(i); 
            matchingSpeciesCodes{end+1} = detections.SpeciesCode(i); 
            matchingCalls{end+1} = detections.Call(i); 
            matchingStartTimes(end+1) = datenum(datetime(detections.EndTime(i), 'InputFormat', 'MM/dd/yyyy HH:mm:ss.S', 'TimeZone', 'local')); 
            matchingEndTimes(end+1) = datenum(datetime(detections.EndTime(i), 'InputFormat', 'MM/dd/yyyy HH:mm:ss.S', 'TimeZone', 'local')); 
            
            % Find matching station in the CalCOFI stations table
            stationKey = strcat(matchingTransect_ID{end}, '_', matchingStation_ID{end});
            stationMatchIdx = find(strcmp(CalCOFI_stations.Station_key, stationKey));
            matchingCalCOFILatitudes(end+1) = CalCOFI_stations.Lat_dec_(stationMatchIdx);
            matchingCalCOFILongitudes(end+1) = CalCOFI_stations.Lon_dec_(stationMatchIdx);
            
        end
    end
end

% Convert results to table
metadataTable = table(matchingDeployments', matchingTransect_ID', matchingStation_ID', matchingSBLatitudes', matchingSBLongitudes', ...
    matchingCalCOFILatitudes', matchingCalCOFILongitudes', matchingInputFile', matchingSpeciesCodes', matchingCalls', ...
     matchingStartTimes', matchingEndTimes', ...
                      'VariableNames', {'Deployment', 'Transect_ID', 'Station_ID', 'Sonobouy_Latitude', 'Sonobouy_Longitude'...
               'CalCOFI_Latitude', 'CalCOFI_Longitude','InputFile', 'SpeciesCode', 'Call', 'StartTime', 'EndTime'});
                

% Define the full path to the output file
outputFile = fullfile(outDir, 'CC201907.csv');
writetable(metadataTable, outputFile)
               



                
                