function normalizedX = normalizeFeature(originx)

meanX = mean(originx);
stdX = std(originx);
normalizedX = originx;

for i = 1:length(meanX)
    for j = 1 : length(originx)
        normalizedX(j,i) = normalizedX(j,i) - meanX(i);
        normalizedX(j,i) = normalizedX(j,i) ./ stdX(i);
    end
end