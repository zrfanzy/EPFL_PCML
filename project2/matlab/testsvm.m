confusion_matrix = zeros(4, 4);
errornum = 0;
errorlist = [];

% testdata range
range = 4801:6000;

% clear temp-use variables
clear tag1 tag2 tag3 tag4;

for i = range
    bestscore = 1;
    
    % get 1vall score
    scoreova = zeros(4, 1);
    for j = 1 : 4
        scoreova(j) = wova(:,j)' * ([train.X_cnn(i,:)]') + bova(j);
        if (scoreova(j) > scoreova(bestscore))
            bestscore = j;
        end
    end
    
    % get 1v1 score
    scoreovo = zeros(6, 1);
    vote = zeros(4, 1);
    for j = 1 : 6
        scoreovo(j) = wovo(:,j)' * ([train.X_cnn(i,:)]') + bovo(j);
        if (scoreovo(j) > 0)
            votefor = vslist(1, j);
        else
            votefor = vslist(2, j);
        end
        vote(votefor) = vote(votefor) + 1;
    end
    
    confusion_matrix(train.y(i), bestscore) = confusion_matrix(train.y(i), bestscore) + 1;
    if ~(bestscore == train.y(i))
        errornum = errornum + 1;
        errorlist = [errorlist,i];
    end
end

% clear temp-use variables
clear bestscore i j range scoreova scoreovo y 