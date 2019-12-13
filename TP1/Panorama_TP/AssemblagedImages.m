%%% Script - Assemblage d'Images
% Question 2 du TP-1
% Autheurs : Gabriel H. Riqueti et Victor K. Nascimento Kobayashi
% Orientateurs : Antoine Manzanera et David Filliat
close;
clc;

% Réglages
nImgs = 3; % nombre des images assemblées : 2 ou 3
nPts  = 4;  % nombre de points considérés pour exécuter l'hommographie

% Chargement de l'image
ima = double(imread('Amst-1.jpg'))/255;
imb = double(imread('Amst-2.jpg'))/255;
imc = double(imread('Amst-3.jpg'))/255;

% Détermination de la taile des images assemblées
[h,w] = size(imb);
xmin = -7*w/20;
if nImgs == 2
    xmax = 11*w/32;
    ymin = -h/10;
    ymax = 33*h/32;
elseif nImgs == 3
    xmax = 11*w/14;
    ymin = -h/3;
    ymax = 7*h/6;
end

% Points d'origine et de destin à résoudre
if nPts == 4
    Pta  = [[435;136;1],[471;135;1],[494;246;1],[466;247;1]];
    Ptab = [[1;136;1],  [39;136;1], [60;248;1], [31;249;1]];
else
    Pta  = [[435;136;1],[471;135;1],[494;246;1],[466;247;1],[481;237;1],[440;157;1],[440;184;1]];
    Ptab = [[1;136;1],  [39;136;1], [60;248;1], [31;249;1], [45;238;1], [6;158;1],  [6;185;1]];
end
Ptc  = [[19;142;1], [57;142;1], [66;215;1], [36;217;1]];
Ptcb = [[400;144;1],[436;143;1],[446;216;1],[417;217;1]];

% Estimation de l'homographie par Direct Linear Transformation
Ha = homography2d(Pta,Ptab);
Hc = homography2d(Ptc,Ptcb);

bbox = [xmin, xmax, ymin, ymax];

% Application des l'hommographies
ima_warped = vgg_warp_H(ima, Ha, 'linear', bbox);
imb_warped = vgg_warp_H(imb, eye(8), 'linear', bbox);
if nImgs == 3
    imc_warped = vgg_warp_H(imc, Hc, 'linear', bbox);
end

% Présentation du résultat
im_fused = max(ima_warped, imb_warped);
if nImgs == 2
    imwrite(im_fused,'Amst-12.JPG');
    imagesc(im_fused);
elseif nImgs == 3
    im_fused = max(im_fused, imc_warped);
    imwrite(im_fused,'Amst-123.JPG');
    imagesc(im_fused);
end

