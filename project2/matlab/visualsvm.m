W = reshape(w, 13, 13, 32);
normalizedW = zeros(13, 13, 32);
sumW = zeros(13, 13);
maxW = zeros(13, 13);
for i = 1 : 13
    for j = 1 : 13
        maxW(i,j) = max(W(i,j,:));
        for k = 1 : 32
            if W(i,j,k) > 0
                sumW(i,j) = sumW(i,j) + W(i,j,k);
            end
        end
        for k = 1 : 32
            if W(i,j,k) > 0
                normalizedW(i,j,k) = W(i,j,k);%/nowsum;
            else
                normalizedW(i,j,k) = 0;
            end
        end
    end
end
normalizedW = single(normalizedW(:, :, 1:31));
%imhog=vl_hog('render', normalizedW ,'verbose');
%clf ; imagesc(imhog) ; colormap gray ;
clf; imagesc(maxW); colormap gray;