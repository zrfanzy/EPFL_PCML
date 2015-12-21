fileID = fopen('logs','r');
formatSpec = '%d\n';
A = fscanf(fileID,formatSpec)
% save('epoch_train_cnn512_dropout95.mat','A')

train = load('epoch_train_cnn512_dropout5.mat'); 
train = train.A;
test = load('epoch_test_cnn512_dropout5.mat'); 
test = test.A;
train = 100*ones(length(train), 1) - train;
test = 100*ones(length(test), 1) - test;
train5 = train;
test5 = test;

clear train;
clear test;

train = load('epoch_train_cnn512_dropout6.mat'); 
train = train.A;
test = load('epoch_test_cnn512_dropout6.mat'); 
test = test.A;
train = 100*ones(length(train), 1) - train;
test = 100*ones(length(test), 1) - test;
train6 = train;
test6 = test;

clear train;
clear test;


train = load('epoch_train_cnn512_dropout8.mat'); 
train = train.A;
test = load('epoch_test_cnn512_dropout8.mat'); 
test = test.A;
train = 100*ones(length(train), 1) - train;
test = 100*ones(length(test), 1) - test;
train8 = train;
test8 = test;

clear train;
clear test;
train = load('epoch_train_cnn512_dropout95.mat'); 
train = train.A;
test = load('epoch_test_cnn512_dropout95.mat'); 
test = test.A;
train = 100*ones(length(train), 1) - train;
test = 100*ones(length(test), 1) - test;
train95 = train;
test95 = test;

clear train;
clear test;

train = load('epoch_train_cnn512_nodropout.mat'); 
train = train.A;
test = load('epoch_test_cnn512_nodropout.mat'); 
test = test.A;
train = 100*ones(length(train), 1) - train;
test = 100*ones(length(test), 1) - test;

% semilogy(1:length(train1),0.01.*train1);hold on;
% semilogy(1:length(test1),0.01.*test1);hold on;
% semilogy(1:length(train),0.01.*train);hold on;
% semilogy(1:length(test),0.01.*test);
plot(1:length(train1)-1,0.01.*train6(2:end));hold on;
plot(1:length(test1)-1,0.01.*test6(2:end));hold on;
plot(1:length(train)-1,0.01.*train8(2:end));hold on;
plot(1:length(test)-1,0.01.*test8(2:end));hold on;
plot(1:length(train)-1,0.01.*train95(2:end));hold on;
plot(1:length(test)-1,0.01.*test95(2:end));hold on;
plot(1:length(train)-1,0.01.*train(2:end),'LineWidth',2);hold on;
plot(1:length(test)-1,0.01.*test(2:end),'LineWidth',2);hold on;
% 

% legend('train error of ');
legend({'trI', 'teI',...
    'trII', 'teII',...
    'trIII', 'teIII',...
    'trIV','teIV'},'Location','best','Box','off');
grid on;
% set(gca,'xticklabel',[],'yticklabel',[]);
hx = xlabel('epoch');
hy = ylabel('error');

% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',17,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.3*[1 1 1],'ycolor',0.3*[1 1 1]);
set([hx; hy],'fontsize',16,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

print -dpdf epoch.pdf