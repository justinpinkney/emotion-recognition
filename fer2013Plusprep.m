dataRoot = '.';

imageTable = readtable('fer2013.csv', 'Delimiter', ',');
labels = readtable('fer2013new.csv');
emotions = categorical(labels.Properties.VariableNames(3:end));
[~, idx] = max(labels{:, 3:end}, [], 2);
label_data = table(categorical(labels.Usage), ...
                    labels.ImageName, ...
                    emotions(idx)', ...
                    'VariableNames', {'Dataset', 'filename', 'emotion'});
                
for emotion = emotions
    if ~isfolder(char(emotion))
        mkdir(char(emotion))
    end
end

parfor iRow = 1:height(labels)
    filename = labels.ImageName{iRow};
    if isempty(filename)
        continue;
    end
    [~, emotionIdx] = max(labels{iRow, 3:end});
    if emotionIdx >= 9
        continue;
    end

    votes = labels{iRow, 3:end};
    if votes(emotionIdx) < 5
        continue;
    end
    folderName = char(emotions(emotionIdx));
    
    imageData = imageTable.pixels{iRow};
    im = str2double(split(imageData));
    im = uint8(reshape(im, 48, 48));
    im = im';
    disp(['saving ', filename])
    imwrite(im, fullfile(folderName, filename));
end

% label_data(strcmp(label_data.filename, ''), :) = [];
% 
% images = imageDatastore(dataRoot, 'IncludeSubfolders', true);
%%
% for iFile = 1:numel(images.Files)
%     x = images.Files{iFile};
%     if contains(x,'Training')
%         continue
%     end
%     if ~strcmp(label_data.filename, getFilename(x))
%         disp(x)
%     else
%         imshow(images.Files{iFile})
%         title(char(label_data.emotion(strcmp(label_data.filename, getFilename(x)))))
%         pause(1)
%     end
% end
% relevant_labels = cellfun(@(x) label_data.emotion(strcmp(label_data.filename, getFilename(x))), ...
%                             images.Files, 'UniformOutput', true);
% images.Labels = label_data;
