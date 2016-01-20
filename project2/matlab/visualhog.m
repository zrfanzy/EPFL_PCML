function hog = visualhog(p)

load('../train.mat');
x = train.X_hog(p,:);
x = reshape(x,13,13,32);
x = x(:,:,1:31);
imhog=vl_hog('render',x,'verbose');
clf ; imagesc(imhog) ; colormap gray ;
hog = imhog;