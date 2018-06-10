results = emotionNet.classify(test);
nSide = 5;
indexes = randi(numel(test.Files), nSide^2, 1);
figure
for iImage = 1:nSide^2
    thisIndex = indexes(iImage);
    subplot(nSide, nSide, iImage)
    imshow(test.readimage(thisIndex));
    xlabel(char(results(thisIndex)))
end