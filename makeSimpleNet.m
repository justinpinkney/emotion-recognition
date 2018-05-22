function layers = makeSimpleNet(imageSize, classes)

    layers =[
        imageInputLayer(imageSize)
        convolution2dLayer([3, 3], 16, 'Padding', 'same')
        reluLayer()
        batchNormalizationLayer()
        maxPooling2dLayer([2, 2], 'Stride', [2, 2])
        convolution2dLayer([3, 3], 32, 'Padding', 'same')
        reluLayer()
        batchNormalizationLayer()
        maxPooling2dLayer([2, 2], 'Stride', [2, 2])
        fullyConnectedLayer(classes)
        softmaxLayer()
        classificationLayer()
        ];
end
