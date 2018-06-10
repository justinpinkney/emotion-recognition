faceDetector = vision.CascadeObjectDetector;
I = imread('visionteam.jpg');
bboxes = faceDetector(I);
IFaces = I;

for iBox = 1:size(bboxes,1)
    thisbox = bboxes(iBox, :);
    im = imcrop(I, thisbox);
    inputIm = imresize(rgb2gray(im), [48, 48]);
    label = net.classify(inputIm);
    a = squeeze(net.activations(inputIm, 'output_softmax'));
    figure
    subplot(1,2,1)
    bar(a)
    subplot(1,2,2)
    imshow(im)
    
    IFaces = insertObjectAnnotation(IFaces,'rectangle',thisbox,char(label));   
end

figure
imshow(IFaces)
title('Detected emotions');

%%
% camera = webcam;
figure
keepRolling = true;
set(gcf,'CloseRequestFcn','keepRolling = false; closereq');

while keepRolling
    im = snapshot(camera);
    image(im)
    bboxes = faceDetector(im);
    imface = im;
    if ~isempty(bboxes)
        imface = imcrop(im, bboxes(1,:));
        im = insertObjectAnnotation(im,'rectangle',bboxes(1,:),'face');   
    end
    inputIm = imresize(rgb2gray(imface), [48, 48]);
    label = net.classify(inputIm);
    a = squeeze(net.activations(inputIm, 'output_softmax'));
%     figure
    subplot(1,2,1)
    ax = bar(a);
    ax.Parent.XTickLabel = cellstr(net.Layers(end).ClassNames);
    ax.Parent.XTickLabelRotation = 90;
    subplot(1,2,2)
    imshow(im)
    title(char(label))
    drawnow
end