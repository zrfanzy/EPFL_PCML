function normalizedX = normalizeFeaturebyOther(originx, normalx)

meanX = mean(originx);
stdX = std(originx);
normalizedX = normalx;

for i = 1:length(meanX)
    for j = 1 : length(normalx)
        normalizedX(j,i) = normalizedX(j,i) - meanX(i);
        normalizedX(j,i) = normalizedX(j,i) ./ stdX(i);
    end
end