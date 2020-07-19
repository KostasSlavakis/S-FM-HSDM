function PARAM = l1scenario(s)
    
    if ( s == 1 )

        PARAM.UseAR = 'no';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 2e-3;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 6;
        
        % PARAM.NOEXP = 500;
        PARAM.NOEXP = 1;
        % PARAM.NoIter = 5e4;
        PARAM.NoIter = 5e3;
        PARAM.D = 100;
        PARAM.SparsityPerc = 1/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-1;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-1;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-1;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 5e2;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-2;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e1;
        PARAM.rhoADMM = 1e1;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 5e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.9;
        PARAM.gammaI = 1e0;
        PARAM.LipSFMI = 90e-4;
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.9;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 90e-4;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;

    elseif ( s == 2 )
        
        PARAM.UseAR = 'no';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 2e-3;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 6;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-1;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-1;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-1;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 5e2;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-2;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e1;
        PARAM.rhoADMM = 1e1;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 5e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 1e2;
        PARAM.LipSFMI = 7e-3;        
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 9e-3;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;
        
    elseif ( s == 3 )
        
        PARAM.UseAR = 'no';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 5e-3;
        PARAM.SNRdB = 10;
        PARAM.SNRdBAR = 6;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 1/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e3;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-2;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e1;
        PARAM.rhoADMM = 1e1;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 5e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.9;
        PARAM.gammaI = 1e0;
        PARAM.LipSFMI = 10e-3;
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.9;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 10e-3;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;
        
    elseif ( s == 4 )
        
        PARAM.UseAR = 'no';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 5e-3;
        PARAM.SNRdB = 10;
        PARAM.SNRdBAR = 6;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e3;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-2;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e1;
        PARAM.rhoADMM = 1e1;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 1e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 5e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 5e2;
        PARAM.LipSFMI = 20e-4;
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 40e-4;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;

    elseif ( s == 5 )

        PARAM.UseAR = 'yes';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 25e-4;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 5;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 1/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e3;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-3;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e2;
        PARAM.rhoADMM = 1e2;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (2e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 1e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.9;
        PARAM.gammaI = 5e2;
        PARAM.LipSFMI = 65e-4;        
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.9;
        PARAM.muSFMIV = 1e0;
        PARAM.LipSFMIV = 30e-4;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;

    elseif ( s == 6 )

        PARAM.UseAR = 'yes';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 40e-4;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 5;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e3;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);

        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-3;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e2;
        PARAM.rhoADMM = 1e2;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (2e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 1e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 5e2;
        PARAM.LipSFMI = 60e-4;        
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.6;
        PARAM.muSFMIV = 1e0;
        PARAM.LipSFMIV = 22e-4;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;

    elseif ( s == 7 )
        
        PARAM.UseAR = 'yes';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 10e-3;
        PARAM.SNRdB = 10;
        PARAM.SNRdBAR = 5;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 1/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e3;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);

        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-3;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e2;
        PARAM.rhoADMM = 1e2;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (5e-5)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 1e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.9;
        PARAM.gammaI = 5e2;
        PARAM.LipSFMI = 50e-4;
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 20e-4;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;
        
    elseif ( s == 8 )
        
        PARAM.UseAR = 'yes';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 10e-3;
        PARAM.SNRdB = 10;
        PARAM.SNRdBAR = 5;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e3;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e3;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);

        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-3;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e2;
        PARAM.rhoADMM = 1e2;
        
        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 1e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 5e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 5e2;
        PARAM.LipSFMI = 20e-4;
        % Standard loss.
        PARAM.alphaII = 0.9;
        PARAM.LipSFMII = 1e-1;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 5e-4;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;
        
    elseif ( s == 9 )
        
        PARAM.UseAR = 'no';
        PARAM.UseNoiseInModel = 'no';
        PARAM.WeightLossReg = 1e-20;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 5;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 1e2;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e-1;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 1e2;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e-1;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-2;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e1;
        PARAM.rhoADMM = 1e1;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-2)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 1e-1;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 1e1;
        PARAM.LipSFMI = 40e-3;
        % Standard loss.
        PARAM.alphaII = 0.5;
        PARAM.LipSFMII = 40e-3;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 30e-3;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;

    elseif ( s == 10 )
        
        PARAM.UseAR = 'yes';
        PARAM.UseNoiseInModel = 'no';
        PARAM.WeightLossReg = 1e-20;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 5;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e4;
        PARAM.D = 100;
        PARAM.SparsityPerc = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 1e2;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e-1;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 1e2;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e-1;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-3;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e2;
        PARAM.rhoADMM = 1e2;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-4);
        PARAM.ConstASVB = 5e-6;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-2)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 1e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 1e0;
        PARAM.LipSFMI = 30e-3;
        % Standard loss.
        PARAM.alphaII = 0.5;
        PARAM.LipSFMII = 30e-3;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 9e-3;
        PARAM.ConstAboveEigenV = 1;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;
        
    elseif ( s == 11 )
        
        PARAM.UseAR = 'no';
        PARAM.UseNoiseInModel = 'yes';
        PARAM.WeightLossReg = 5e-3;
        PARAM.SNRdB = 20;
        PARAM.SNRdBAR = 6;
        
        PARAM.NOEXP = 500;
        PARAM.NoIter = 5e3;
        PARAM.SwitchPoint = 2.5e3;
        PARAM.D = 100;
        PARAM.SparsityPercA = 1/100;
        PARAM.SparsityPercB = 10/100;
        % PARAM.VarNonZeroValuesTarget = .1;     
        
        PARAM.delta = .9925;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RLS.
        PARAM.epsilonRLS = 1e-2;
        
        % l0-RLS.
        PARAM.deltal0RLS = 1e-2;
        PARAM.betal0RLS = 5e1;
        PARAM.UseConstantGammal0RLS = 'yes';
        PARAM.ConstantGammal0RLS = 1e2;
        PARAM.Multiplygammal0RLSby = 1;
        PARAM.lambdal0RLS = 1-(1e-5);
        
        % l1-RLS.
        PARAM.deltal1RLS = 1e-2;
        PARAM.betal1RLS = 5e1;
        PARAM.UseConstantGammal1RLS = 'yes';
        PARAM.ConstantGammal1RLS = 1e2;
        PARAM.Multiplygammal1RLSby = 1;
        PARAM.lambdal1RLS = 1-(1e-5);
        
        % Prox-SVRG.
        PARAM.mSVRG = 100;
        PARAM.etaSVRG = 1e-2;

        % SVRG-ADMM.
        PARAM.mADMM = 5;
        PARAM.bADMM = 100;
        PARAM.etaADMM = 1e1;
        PARAM.rhoADMM = 1e1;

        % ASVB.
        PARAM.lambdaASVB = 1-(1e-2);
        PARAM.ConstASVB = 1e-7;
        
        PARAM.ARcoeff = sqrt(1-10^(-PARAM.SNRdBAR/10));
        if (strcmp(PARAM.UseAR,'yes'))
            PARAM.LipCoeff = 1/(1-PARAM.ARcoeff^2)+(1e-2);
        else
            PARAM.LipCoeff = 1+(1e-2);
        end

        % AC-SA. See (33):
        % PARAM.GammaACSA = min(1e-4,1/2/PARAM.LipCoeff);
        PARAM.GammaACSA = (1e-4)/2/PARAM.LipCoeff;
        
        % SDA. See (6).
        PARAM.gammaSDA = 5e-2;
        PARAM.lambdaSDA = PARAM.WeightLossReg;
        
        % S-FM-HSDM.
        % Adaptive constraints (T = Prox).
        PARAM.alphaI = 0.5;
        PARAM.gammaI = 1e0;
        PARAM.LipSFMI = 1500e-4;
        % Standard loss.
        PARAM.alphaII = 0.5;
        PARAM.LipSFMII = 5e-2;
        % Adaptive constraints (T = Grad).
        PARAM.alphaIV = 0.5;
        PARAM.muSFMIV = 1;
        PARAM.LipSFMIV = 2000e-4;
        PARAM.ConstAboveEigenV = 5;
        % Adaptive constraints (T = Proj).
        % PARAM.LipSFMIII = PARAM.LipSFM;
        
    end

end
