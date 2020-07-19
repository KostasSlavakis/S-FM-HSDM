% Scenarios = [1:10];
Scenarios = [1];

% parfor j=1:(length(Scenarios))
for j=1:(length(Scenarios))
    % l1RLSforgetsNoDisplay(Scenarios(j));
    l1RLSforgets(Scenarios(j));
end