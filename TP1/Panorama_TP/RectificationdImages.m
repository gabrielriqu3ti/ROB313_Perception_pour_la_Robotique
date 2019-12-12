%%% Script - Rectification d'Images
% Question 1 du TP-1
% Autheurs : Gabriel H. Riqueti et Victor K. Nascimento Kobayashi
% Orientateurs : Antoine Manzanera et David Filliat
close;
clc;

% Chargement de l'image
imrgb = double(imread('Pompei.jpg'))/255;
%imagesc(imrgb);

% Points d'origine et de destin à résoudre
PtO = [[143;43;1],[111;256;1],[356;43;1],[371;260;1]];
PtD = [[143;43;1],[143;256;1],[356;43;1],[356;256;1]];

% Estimation de l'homographie par Direct Linear Transformation
H = homography2d(PtO,PtD);

% Présentation du résultat
imagesc(vgg_warp_H(imrgb,H));
imwrite(vgg_warp_H(imrgb,H),'Pompei_Carre.JPG');