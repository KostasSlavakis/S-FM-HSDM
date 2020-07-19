% load l1RLSresultsScenario11.mat;

NoIter = PARAM.NoIter;
colororder = [0.76  0.57  0.17
              0.54  0.63  0.22
              0.34  0.57  0.92
              1.00  0.10  0.60
              0.10  0.49  0.47
              0.66  0.34  0.65
              0.99  0.41  0.23
              0.00  0.00  1.00
              0.25  0.25  0.25
              1.00  0.00  0.00 
              0.00  0.75  0.75
              0.75  0.00  0.75
              0.75  0.75  0.00 
              0.75  0.25  0.25
              0.95  0.95  0.00 
              0.25  0.25  0.75
              0.75  0.75  0.75
              0.00  0.50  0.00 
              0.88  0.75  0.73
              0.00  1.00  0.00];
markers = {'+','o','*','x','s','d','^','v','>','<','p','h','.'};

Deviations = {DevOptRLSav, DevOptOSCDav, DevOptOCCDav, DevOptl0RLSav, DevOptl1RLSav, ...
              DevOptSVRGav, DevOptADMMav, DevOptASVBav, DevOptACSAav, ...
              DevOptSDAav, DevOptSFMIIav, DevOptSFMIVav, DevOptSFMIav};
DevNames = {'RLS', 'RLS-OSCD-TWL', 'RLS-OCCD-TWL', 'l0-RLS', 'l1-RLS', ...
            'Prox-SVRG', 'SVRG-ADMM', 'ASVB-MPL', 'AC-SA', ...
            'SDA', 'H-RLS (T=consensus)', 'H-RLS (T=Grad)', 'H-RLS (T=Prox)'};

% MarkerIndices = [1:2:10, 2e1:2e1:1e2, 2e2:2e2:1e3, 2e3:1e3:5e3];
MarkerIndices = [1:2:10, 2e1:2e1:1e2, 2e2:2e2:1e3, 2e3:1e3:5e3];

FigHandle = figure('Position', [2000, 750, 1000, 500]);
% subplot(1,2,1);
hold all;
for jj = 1:length(Deviations)
    Dev = Deviations{jj};
    plot([1:NoIter], Dev, 'linestyle', '-', 'linewidth', 3, ...
         'Color', colororder(jj,:));
    plot(MarkerIndices, Dev(MarkerIndices), 'linestyle', 'none', ...
         'Marker', markers{jj}, 'MarkerSize', 15, 'Color', ...
         colororder(jj,:), 'linewidth', 2);
    LH(jj) = plot(nan, nan, 'linestyle', '-', 'Marker', ...
                  markers{jj}, 'linewidth', 2, 'MarkerSize', ...
                  15, 'Color', colororder(jj,:));
    L{jj} = DevNames{jj};
end
% set(gca,'xscale','log');
set(gca,'yscale','log');
set(gca,'FontSize',16);
FigLegend = legend(LH,L,'Location','BestOutside');
set(FigLegend,'FontSize',16);
axis([1 NoIter -inf 5]);
ylabel('Normalized deviation from actual system');
xlabel('Number of iterations');
% axis([1 NoIter 10^(-10) 2]);
box on;
hold off;

% subplot(1,2,2);
% hold all;
% for jj = 1:length(Losses)
%     Loss = Losses{jj};
%     plot([1:NoIter], Loss, 'linestyle', '-', 'linewidth', 3, ...
%          'Color', colororder(jj,:));
%     plot(MarkerIndices, Loss(MarkerIndices), 'linestyle', 'none', ...
%          'Marker', markers{jj}, 'MarkerSize', 15, 'Color', ...
%          colororder(jj,:), 'linewidth', 2);
%     LH(jj) = plot(nan, nan, 'linestyle', '-', 'Marker', ...
%                   markers{jj}, 'linewidth', 2, 'MarkerSize', ...
%                   15, 'Color', colororder(jj,:));
%     L{jj} = LossNames{jj};
% end
% set(gca,'xscale','log');
% set(gca,'yscale','log');
% plot([1,NoIter],[.5*param.lmin,.5*param.lmin], '-k');
% FigLegend = legend(LH,L,'Location','southeast');
% set(FigLegend,'FontSize',10);
% axis([1 NoIter param.lmin/10 10*param.lmin]);
% hold off;
