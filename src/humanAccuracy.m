dataRoot = 'data';

% TODO assert the data files are present

labels = readtable(fullfile(dataRoot, 'fer2013new.csv'));
emotions = categorical(labels.Properties.VariableNames(3:end));
[~, idx] = max(labels{:, 3:end}, [], 2);
label_data = table(categorical(labels.Usage), ...
                    labels.ImageName, ...
                    emotions(idx)', ...
                    'VariableNames', {'Dataset', 'filename', 'emotion'});


labels(~strcmp(labels.Usage, 'PrivateTest'), :) = [];
count = 1;
for iRow = 1:height(labels)
    thisRow = labels(iRow, :);
    
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
    maxVotes = votes(emotionIdx);
    if maxVotes < 6
        continue;
    end
    certainty(count) = maxVotes/sum(votes);
    count = count + 1;
end
