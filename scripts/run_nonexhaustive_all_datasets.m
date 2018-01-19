%=====Set parameters and Load data====
clear;
startup;

fpath = 'D:\Google Drive\Studies\Research\data\';
fpath = GetFullPath(fpath);
datasets = {
    'flow_cytometry\ndd_g1',{[],[2;6],[2;6;3;1],[2;6;3;1;7;4],1:7},5
    'flight_line\FlightlineC1Data',{[],[8 7],[8;7;6;2],[8;7;6;2;4;1],1:9},10
    'vehicle\shuttle',{[],[1 4], [1 4 5 3 2],1:7},1
    'segmentation\segmentation',{[],[1 2],[1 2 4 6],1:7},1
    'letter\letter',{[],1:6:26,1:3:26,[1:3:26 2:6:26],1:26},10
    
    'stemcell\stemcell_1',{[],[4 5],[4 5 3 2]},5
    'uci_har\uci_har',{[],[1 2],[1 2 3 6],1:6},1
    'sat\sat',{[],[1 7],[1 7 3 5],1:7},1
    'indian_pines\indian_pines',{[],[11,2,14,10],[11,2,14,10,3,6,12,5],[11,2,14,10,3,6,12,5,8,15,4,13],1:16},5
    'vowel\vowel',{[],1:4,1:8,1:11},1
    
    'FlowCapIII\FlowCapIII_LA_4_SI_2_formated',{1},1
    'CRISM\tpami_9971_xy',{[2, 3, 10, 14, 26, 34]+1},5
    };

for name_idx = [11 12]
% for name_idx = [31:33 35:40]
name = datasets{name_idx,1};
fprintf('\nRunning data set %s\n', name);
load([fpath name '.mat'],'x','y');
has0 = ~all(y(:,1));
y(:,1) = y(:,1)+has0;
[nuy,uy] = hist(y(:,1),unique(y(:,1)));
[~,idx] = sort(nuy,'descend');
ins = datasets{name_idx, 2};
if isempty(ins) % Add training class by size
    ins = cell(ceil(numel(uy)/2)+1,1);
    for nk = 2:numel(ins)
        ins{nk} = uy(idx(1:(nk-1)*2));
    end
end
if size(x,2)>50
    x = pca_wzy(x,30);
end
x = standardize(x);
d = size(x,2);

%-----model control parameters-----
max_sweep = 10;
init_sweep = 10;
burnin = 0;
sample = 2;
n_iter = 3;

%-----hyper params------
m = d+2; c_s = (m-d-1);
mu0 = zeros(1,d); sigma0 = eye(d) * c_s;
kappa0 = 0.1; kappa1 = 0.5;
alpha = 1; gamma = 1;
priors{1} = [mu0;sigma0];
confs{1} = [m kappa0 alpha];
priors{2} = [mu0;sigma0]; specs{2} = [ 0 0 0 [0 0] 0 0 0];
confs{2} = [m kappa0 kappa1 alpha gamma specs{2}];

c1 = 0.1; c2 = d+2;
beta0 = d; alpha0 = beta0*kappa0 + 1;
beta1 = d; alpha1 = beta1*kappa1 + 1;
priors{3} = [mu0;sigma0/(c2-d)]; specs{3} = [ 0 0 0 [1 0] 0 0 1];
confs{3} = [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma specs{3}];
priors{4} = priors{3}; specs{4} = [5 0 0 [0 0] 4 0]; % ratio, prior, train, kap1>kap0, kap1<x*kap0, all in H, table, weighted kappa
confs{4} = [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma specs{4}];
priors{5} = priors{3}; specs{5} = [5 0 0 [1 0] 4 0]; % ratio, prior, train, kap1>kap0, kap1<x*kap0, all in H, table
confs{5} = [m c1 c2 alpha0 beta0 alpha1 beta1 alpha gamma specs{5}];

models = [1 2 3 3 3];% 1: igmm 2: i2gmm 3: i3gmm
conf_names = cell(1,numel(confs));
legends = cell(2, numel(confs));
for i = 1:numel(confs)
    confstr = [];
    if models(i)>1
        conf = confs{i};
        if sum(specs{i}) > 0
            confstr = num2str(specs{i},'%d');
        end
    end
    conf_names{i} = [sprintf('i%dgmm',models(i)) confstr];
    legends{1,i} = ['mean ' conf_names{i}];
    legends{2,i} = ['weighted ' conf_names{i}];
end

h1 = figure;
h2 = figure; % test gt
h3 = figure; % test
h4 = figure;
movegui(h1, 'northwest');
movegui(h2, 'north');
movegui(h3, 'southwest');
movegui(h4, 'south');
cm = lines(numel(confs));
F1_stats = zeros(numel(ins), numel(confs) * 4);
Acc_stats = zeros(numel(ins), numel(confs) * 4);
Time_stats = zeros(numel(ins), numel(confs) * 2);
conf_idxs = [3 2 1];
for conf_idx = conf_idxs
    model = models(conf_idx);
    fprintf('Running %s\n',conf_names{conf_idx});
    conf = confs{conf_idx};
    prior = priors{conf_idx};
    for j = 1:numel(ins)
        %------split data into training and testing------
        if datasets{name_idx,3} == 1
            in = ismember(y(:,1), ins{j}) & y(:,2)==1;
        else
            in = ismember(y(:,1), ins{j}) & (mod(1:length(y), datasets{name_idx,3})==0)';
        end
        xtrain = x(in, :);
        ytrain = y(in, 1);
        xtest = x(~in, :);
        ytest = y(~in, 1);
        ntrain = size(xtrain,1);
        y1 = [ytrain;ytest];
        if isempty(xtrain) || numel(unique(ytrain))<2
            x1 = LDA_w([xtrain;xtest],y1);
        else
            x1 = LDA_w(xtrain,ytrain,xtest);
        end
        fprintf('Training classes: ');
        fprintf('%d ', numel(unique(ytrain))); fprintf('\n');
        
        fname = [fpath name '_' conf_names{conf_idx} '_' num2str(numel(ytrain)) '\'];
        save_data([fname 'data'], xtest, conf, prior, [ytrain xtrain], ytest); % For exe
        clr = hsv(max(y1));
        figure(h1); clf(h1); % test gt
        gscatter(x1(1+ntrain:end,1),x1(1+ntrain:end,2), ytest, clr(unique(ytest),:));
        h1xlim = get(gca,'xlim');
        h1ylim = get(gca,'ylim');
        figure(h2); clf(h2); % train gt
        gscatter(x1(1:ntrain,1),x1(1:ntrain,2), ytrain, clr(unique(ytrain),:));
        xlim(h1xlim);
        ylim(h1ylim);
        drawnow;
        saveas(h1, [fname 'data_test.png']);
        saveas(h2, [fname 'data_train.png']);
        
        %======Run i3gmm======
        F1s = zeros(n_iter, 2);
        Accs = zeros(n_iter, 2);
        Times = zeros(n_iter, 1);
        hyperparams = zeros(d+2,d,n_iter);
        if model > 1
            samples_all = zeros(length(y), 2*sample, n_iter);
        else
            samples_all = zeros(length(y), sample, n_iter);
        end
        for i = 1 : n_iter
            fres = [ '"' fname 'result_m' num2str(m) '_c' num2str(c_s) ...
                '_k0_' num2str(kappa0) '_k1_' num2str(kappa1) '_' num2str(i) '"'];
            % Run exe program
            args = gen_arguments([fname 'data'], fres, model, max_sweep, burnin, sample, init_sweep);
            fid = fopen('cmd.txt', 'wt');
            fprintf(fid, '%s', args);
            fclose(fid);
            tic
            [status, ret] = system(['..\Release\i2gmm_semi.exe ' args]);
            Times(i) = toc;
            if status
                warning('Something wrong when running i2gmm:\n %s',ret);
                continue;
            end
            
            %-------read results---------
            fres = fres(2:end-1);
            samples = readMat([fres '_samplelabels.txt'], 'text')';
            samples_all(:,:,i) = samples;
            if model > 1
                hyperparams = readMat([fres '_hyperparams.txt'],'text');
                labels = samples(:,mod(1:(2*sample),2)==1);
            else
                labels = samples;
            end
            label_last = readMat([fres '_lastlabels.txt'], 'text')';
            label_best = readMat([fres '_bestlabels.txt'], 'text')';
%             likelihood = readMat([fres '_likelihoods.matrix'], 'double');
            %calc_F1(label_best(:,1), y1,2,ntrain,1)
            
            %-------align labels--------
            unique_y_s = zeros(sample,1);
            for s_idx = 1:sample
                unique_y_s(s_idx) = numel(unique(labels(:,s_idx)));
            end
            [~,ref_idx] = max(flip(unique_y_s));
            ref_idx = sample-ref_idx+1;
            label = align_labels(labels, labels(:,ref_idx), ntrain);
            label = align_label(y1,label,ntrain);
%             label = align_labels(labels, y1, ntrain);

%             F1s(i,:) = calc_F1(label, y1, 2, ntrain);
%             F1s(i,:) = calc_F1(label, y1, 2);
            if has0
                F1s(i,:) = calc_F1(label(y1~=1), y1(y1~=1), 2, ntrain,1);
            else
                F1s(i,:) = calc_F1(label, y1, 2, ntrain,1);
            end
            Accs(i,:) = calc_acc(label((1+ntrain:end),:),ytest);
            fprintf('mean F1: %.4f, Weighted F1: %.4f, ', F1s(i, :));
            fprintf('Weighted Acc: %.4f, mean Acc: %.4f, ', Accs(i, :));
            fprintf('# class found: %d\n', numel(unique(label)));
            
            clr = hsv(max([y1; label]));
            figure(h1); clf(h1); % test gt
            gscatter(x1(1+ntrain:end,1),x1(1+ntrain:end,2), ytest, clr(unique(ytest),:));
            h1xlim = get(gca,'xlim');
            h1ylim = get(gca,'ylim');
            figure(h2); clf(h2); % train gt
            gscatter(x1(1:ntrain,1),x1(1:ntrain,2), ytrain, clr(unique(ytrain),:));
            xlim(h1xlim);
            ylim(h1ylim);
            figure(h3); clf(h3); % test
            gscatter(x1(1+ntrain:end,1),x1(1+ntrain:end,2), label(1+ntrain:end), clr(unique(label(1+ntrain:end)),:));
            xlim(h1xlim);
            ylim(h1ylim);
            drawnow;
            saveas(h3, [fres '.png']);
        end
        
        Time_stats(j, 2*(conf_idx-1)+1:2*conf_idx) = [mean(Times) std(Times)];
        F1_stats(j, 4*(conf_idx-1)+1:4*conf_idx) = reshape([mean(F1s); std(F1s)],1,[]);
        Acc_stats(j, 4*(conf_idx-1)+1:4*conf_idx) = reshape([mean(Accs); std(Accs)],1,[]);
        fprintf('mean F1: %.4f ± %.4f, Weighted F1: %.4f ± %.4f\n', F1_stats(j,4*(conf_idx-1)+1:4*conf_idx));
        fprintf('mean Acc: %.4f ± %.4f, Weighted Acc: %.4f ± %.4f\n', Acc_stats(j,4*(conf_idx-1)+1:4*conf_idx));
        fprintf('mean Running time: %.4f ± %.4f\n', Time_stats(j, 2*(conf_idx-1)+1:2*conf_idx));
        % fprintf('Hyper-parameters (median):\n');
        % disp(median(hyperparams,3));
        fres = [fname 'model_' conf_names{conf_idx} '_m' num2str(m) '_s' num2str(c_s)...
            '_kap' num2str(kappa0) '_kapp' num2str(kappa1) '_nsweep' num2str(max_sweep)];
        save([fres '.mat'],'in','F1s','Accs','hyperparams','Times','samples_all');
    end
    
    
    % Plot summary and save results
    fname = [fpath name '_' conf_names{conf_idx} '_m' num2str(m) '_kap'...
        num2str(kappa0) '_kapp' num2str(kappa1)];
    save([fname '.mat'], '-regexp', '^(?!(h1|h2|h3|h4)$).');
    nc = zeros(numel(ins),1);
    for i = 1:numel(ins)
        nc(i) = numel(ins{i});
%         nc(i) = ins{i};
    end
    figure(h4);
    errorbar(nc, F1_stats(:,4*(conf_idx-1)+1), F1_stats(:,4*(conf_idx-1)+2), 'Color', cm(conf_idx,:), 'DisplayName', legends{1,conf_idx});
    hold on
    h = errorbar(nc, F1_stats(:,4*(conf_idx-1)+3), F1_stats(:,4*(conf_idx-1)+4), ':', 'Color', cm(conf_idx,:), 'DisplayName', legends{2,conf_idx});
    h.Bar.LineStyle = 'dotted';
    hold on
    drawnow;
end

% figure(h4);
% for conf_idx = conf_idxs
%     % Plot summary and save results
%     nc = zeros(numel(ins),1);
%     for i = 1:numel(ins)
%         nc(i) = numel(ins{i});
%     end
%     errorbar(nc, F1_stats(:,4*(conf_idx-1)+1), F1_stats(:,4*(conf_idx-1)+2), 'Color', cm(conf_idx,:), 'DisplayName', legends{1,conf_idx});
%     hold on
%     h = errorbar(nc, F1_stats(:,4*(conf_idx-1)+3), F1_stats(:,4*(conf_idx-1)+4), ':', 'Color', cm(conf_idx,:), 'DisplayName', legends{2,conf_idx});
%     h.Bar.LineStyle = 'dotted';
%     hold on
% end
hold off
lgd = legend('show');
lgd.Location = 'southeast';
saveas(h4, [fpath name '_m' num2str(m) '_kap'...
        num2str(kappa0) '_kapp' num2str(kappa1) '_F1.png']);
save([fpath name '_m' num2str(m) '_kap'...
        num2str(kappa0) '_kapp' num2str(kappa1) '.mat'], '-regexp', '^(?!(h1|h2|h3|h4|h5)$).');

h5 = figure;
for conf_idx = conf_idxs
    % Plot summary and save results
    nc = zeros(numel(ins),1);
    for i = 1:numel(ins)
        nc(i) = numel(ins{i});
    end
    errorbar(nc, Acc_stats(:,4*(conf_idx-1)+1), Acc_stats(:,4*(conf_idx-1)+2), 'Color', cm(conf_idx,:), 'DisplayName', legends{1,conf_idx});
    hold on
    h = errorbar(nc, Acc_stats(:,4*(conf_idx-1)+3), Acc_stats(:,4*(conf_idx-1)+4), ':', 'Color', cm(conf_idx,:), 'DisplayName', legends{2,conf_idx});
    h.Bar.LineStyle = 'dotted';
    hold on
end
hold off
lgd = legend('show');
lgd.Location = 'southeast';
saveas(h5, [fpath name '_m' num2str(m) '_kap'...
        num2str(kappa0) '_kapp' num2str(kappa1) '_Acc.png']);

close('all');
end