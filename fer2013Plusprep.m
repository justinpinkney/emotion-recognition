dataRoot = 'data';

% TODO assert the data files are present

imageTable = readtable(fullfile(dataRoot, 'fer2013.csv'), 'Delimiter', ',');
labels = readtable(fullfile(dataRoot, 'fer2013new.csv'));
emotions = categorical(labels.Properties.VariableNames(3:end));
[~, idx] = max(labels{:, 3:end}, [], 2);
label_data = table(categorical(labels.Usage), ...
                    labels.ImageName, ...
                    emotions(idx)', ...
                    'VariableNames', {'Dataset', 'filename', 'emotion'});

for usage = unique(labels.Usage)'
    stem = fullfile(dataRoot, usage{1});
    for emotion = emotions
        directory = fullfile(stem, char(emotion));
        if ~isfolder(directory)
            mkdir(directory);
        end
    end
end

parfor iRow = 1:height(labels)
    thisRow = labels(iRow, :);
    thisImageRow = imageTable(iRow, :);
    
    filename = thisRow.ImageName{1};
    if isempty(filename)
        continue;
    end
    [~, emotionIdx] = max(thisRow{1, 3:end});
    if emotionIdx >= 9
        continue;
    end

    votes = thisRow{1, 3:end};
    % Remove uncertain ones (from paper)
    if votes(emotionIdx) < 6
        continue;
    end
    folderName = fullfile(dataRoot, ...
                            thisRow.Usage{1}, ...
                            char(emotions(emotionIdx)));
    
    imageData = thisImageRow.pixels{1};
    im = str2double(split(imageData));
    im = uint8(reshape(im, 48, 48));
    im = im';
    disp(['saving ', filename])
    imwrite(im, fullfile(folderName, filename));
end