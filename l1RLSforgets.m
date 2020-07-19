function l1RLSforgets(Sce)

%% Program created by Kostas Slavakis.

%% Dell server.
% use -noexec matlab
% nohup matlab -nodisplay -nodesktop -nosplash < myfile.m >& Log.out &

    % rng('shuffle');

    %% Problem and parameter definition. Possible scenarios: 1, 2, 3, 4.
    PARAM = l1scenario(Sce);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    NOEXP = PARAM.NOEXP;
    NoIter = PARAM.NoIter;
    D = PARAM.D;
    SparsityPerc = PARAM.SparsityPerc;
    % VarNonZeroValuesTarget = PARAM.VarNonZeroValuesTarget;

    UseAR = PARAM.UseAR;
    UseNoiseInModel = PARAM.UseNoiseInModel;
    WeightLossReg = PARAM.WeightLossReg;
    SNRdB = PARAM.SNRdB;
    SNRdBAR = PARAM.SNRdBAR;

    delta = PARAM.delta;

    ARcoeff = PARAM.ARcoeff;
    LipCoeff = PARAM.LipCoeff;

    epsilonRLS = PARAM.epsilonRLS;

    deltal0RLS = PARAM.deltal0RLS;
    betal0RLS = PARAM.betal0RLS;
    lambdal0RLS = PARAM.lambdal0RLS;
    Multiplygammal0RLSby = PARAM.Multiplygammal0RLSby;
    ConstantGammal0RLS = PARAM.ConstantGammal0RLS;
    UseConstantGammal0RLS = PARAM.UseConstantGammal0RLS;

    deltal1RLS = PARAM.deltal1RLS;
    betal1RLS = PARAM.betal1RLS;
    lambdal1RLS = PARAM.lambdal1RLS;
    Multiplygammal1RLSby = PARAM.Multiplygammal1RLSby;
    ConstantGammal1RLS = PARAM.ConstantGammal1RLS;
    UseConstantGammal1RLS = PARAM.UseConstantGammal1RLS;
    
    mSVRG = PARAM.mSVRG;
    etaSVRG = PARAM.etaSVRG;

    mADMM = PARAM.mADMM;
    bADMM = PARAM.bADMM;
    etaADMM = PARAM.etaADMM;
    rhoADMM = PARAM.rhoADMM;

    lambdaASVB = PARAM.lambdaASVB;
    rhoASVB = PARAM.ConstASVB;
    deltaASVB = PARAM.ConstASVB;
    kappaASVB = PARAM.ConstASVB;
    nuASVB = PARAM.ConstASVB;

    GammaACSA = PARAM.GammaACSA;

    gammaSDA = PARAM.gammaSDA;
    lambdaSDA = PARAM.lambdaSDA;

    alphaI = PARAM.alphaI;
    alphaII = PARAM.alphaII;
    alphaIV = PARAM.alphaIV;
    gammaI = PARAM.gammaI;
    muSFMIV = PARAM.muSFMIV;
    lambdaSFMI = 2*(1-alphaI)/PARAM.LipSFMI - (1e-2);
    lambdaSFMII = 2*(1-alphaII)/PARAM.LipSFMII - (1e-2);
    lambdaSFMIV = 2*(1-alphaIV)/PARAM.LipSFMIV - (1e-2);
    ConstAboveEigenV = PARAM.ConstAboveEigenV;
    % lambdaSFMI = lambdaSFM;
    % lambdaSFMIII = lambdaSFM;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DevOptSFMIav = zeros(1,NoIter);
    DevOptSFMIIav = zeros(1,NoIter);
    % DevOptSFMIIIav = zeros(1,NoIter);
    DevOptSFMIVav = zeros(1,NoIter);
    DevOptRLSav = zeros(1,NoIter);
    DevOptOSCDav = zeros(1,NoIter);
    DevOptOCCDav = zeros(1,NoIter);
    DevOptl0RLSav = zeros(1,NoIter);
    DevOptl1RLSav = zeros(1,NoIter);
    DevOptSVRGav = zeros(1,NoIter);
    DevOptADMMav = zeros(1,NoIter);
    DevOptASVBav = zeros(1,NoIter);
    DevOptACSAav = zeros(1,NoIter);
    DevOptSDAav = zeros(1,NoIter);

    ID = eye(D);

    for noexp=1:NOEXP
        
        DevOptSFMI = zeros(1,NoIter);
        DevOptSFMII = zeros(1,NoIter);
        % DevOptSFMIII = zeros(1,NoIter);
        DevOptSFMIV = zeros(1,NoIter);
        DevOptRLS = zeros(1,NoIter);
        DevOptOSCD = zeros(1,NoIter);
        DevOptOCCD = zeros(1,NoIter);
        DevOptl0RLS = zeros(1,NoIter);
        DevOptl1RLS = zeros(1,NoIter);
        DevOptSVRG = zeros(1,NoIter);
        DevOptADMM = zeros(1,NoIter);
        DevOptASVB = zeros(1,NoIter);
        DevOptACSA = zeros(1,NoIter);
        DevOptSDA = zeros(1,NoIter);
        
        % Pre-processing.
        
        Target = zeros(D,1);
        IndSparseTarget = randperm(D);
        IndSparseTarget = IndSparseTarget(1:ceil(SparsityPerc*D));
        Target(IndSparseTarget) = (1-2*round(rand(ceil(SparsityPerc*D),1))); 
        % + ...
        %    sqrt(VarNonZeroValuesTarget)*randn(ceil(SparsityPerc*D),1);
        % Target(IndSparseTarget) = 1+.5*rand(ceil(SparsityPerc*D),1);
        % Target(IndSparseTarget) =
        % (1-2*round(rand(ceil(SparsityPerc*D),1))).*Target(IndSparseTarget); 
        TargetNorm = norm(Target);
        rhol0RLS = length(find(Target));
        x0 = randn(D,1);
        
        fprintf('\nScenario %d: Experiment %d/%d:\n',Sce,noexp,NOEXP);
        
        a = randn(D,1);

        if (strcmp(UseAR,'yes'))
            NoiseVar = (TargetNorm^2)*(10^(-SNRdB/10))/(1-ARcoeff^2);
        else
            NoiseVar = (TargetNorm^2)*(10^(-SNRdB/10));
        end

        if (strcmp(UseNoiseInModel,'yes'))
            b = a'*Target + sqrt(NoiseVar)*randn;
        else
            b = a'*Target;
        end
        
        r = b*a;
        R = bsxfun(@times,a,a');
        R = (R+R')/2;

        RSFM = R;
        rSFM = r;
        aux = a;
        Resolvent = gammaI*sparse(eye(D)) - gammaI*bsxfun(@times,a,a')/(1/gammaI + a'*a);
        ResolventII = lambdaSFMII*sparse(eye(D)) - ...
            lambdaSFMII*bsxfun(@times,a,a')/(1/lambdaSFMII + a'*a); 
        
        xSFMI = x0;
        Tmap = Resolvent*(xSFMI/gammaI+rSFM);
        TminusGrad = Tmap;
        IminusGrad = xSFMI;
        MidPointSFMI = alphaI*TminusGrad + (1-alphaI)*IminusGrad;
        xSFMI = MidPointSFMI.*(1-lambdaSFMI./max(lambdaSFMI,abs(MidPointSFMI)));
        
        xSFMIIprev = repmat(x0,[1,2]);
        TalphaN = alphaI*repmat(x0,[1,2]) + (1-alphaI)*repmat(x0,[1,2]);
        xSFMIIhalf = TalphaN;
        xSFMIIa = ResolventII*(xSFMIIhalf(:,1)/lambdaSFMII + rSFM);
        xSFMIIb = xSFMIIhalf(:,2).*(1-WeightLossReg*lambdaSFMII./max(WeightLossReg*lambdaSFMII, ...
                                                          abs(xSFMIIhalf(:,2))));
        xSFMII = [xSFMIIa,xSFMIIb];

        % xSFMIII = x0;
        % T0III = xSFMIII - R*(PinvR*xSFMIII) + PinvR*r;
        % % T0III = xSFMIII - BasisR*(BasisR'*xSFMIII) + BasisR*(PseudoEigenValues.*(BasisR'*r));
        % TnAlphaIII = alpha*T0III + (1-alpha)*xSFMIII;
        % xSFMIIIhalf = TnAlphaIII;
        % TnPrevIII = T0III;
        % xSFMIIIprev = xSFMIII;
        % xSFMIII = xSFMIIIhalf.*(1-lambdaSFMIII./max(lambdaSFMIII,abs(xSFMIIIhalf)));
        
        rhoSFMIV = trace(R);
        % rhoSFMIV = eigs(R,1);
        xSFMIVprev = x0;
        TIV = xSFMIVprev - muSFMIV*R*xSFMIVprev/rhoSFMIV + muSFMIV*r/rhoSFMIV;
        xSFMIVmiddle = alphaIV*TIV + (1-alphaIV)*xSFMIVprev;
        xSFMIV = xSFMIVmiddle.*(1-lambdaSFMIV./max(lambdaSFMIV,abs(xSFMIVmiddle)));    
        
        PowerV = randn(D,1);
        % PowerV = PowerV/norm(PowerV);
        Delta = 1;
        DeltaSq = 1;

        %% 
        xRLS = x0;
        Prls = sparse(eye(D))/epsilonRLS;

        rl0RLS = b*a;
        xl0RLS = x0;
        Pl0RLS = sparse(eye(D))/deltal0RLS;
        Gradfl0RLS = zeros(D,1);        
        for i = 1:D
            if ( abs(xl0RLS(i)) <= 1/betal0RLS )
                Gradfl0RLS(i) = betal0RLS*sign(xl0RLS(i)) - (betal0RLS^2)*xl0RLS(i);
            end
        end

        rl1RLS = b*a;
        xl1RLS = x0;
        Pl1RLS = sparse(eye(D))/deltal1RLS;
        Gradfl1RLS = sign(xl1RLS);
        
        xOSCD = x0;
        xOCCD = x0;
        
        %% 
        Buffer4a = [];
        xSVRG = x0;

        xTildeADMM = x0;
        yTildeADMM = x0;
        uTildeADMM = -(1/rhoADMM)*(R*xTildeADMM-r);
        
        %% 
        xASVB = x0;
        Aprev = diag(rand(D,1));
        A = Aprev;
        rASVB = b*a;
        RASVB = bsxfun(@times,a,a');
        dASVB = b^2;
        sigmaASVB = rand(D,1);
        bASVB = (1+kappaASVB)./(nuASVB + .5*rand(D,1));
        
        %% The minimizer x1 of the omega = l2norm function.
        xACSA = zeros(D,1);
        xACSAag = xACSA;
        
        %%
        thetaSDA = x0;
        thetaAvSDA = zeros(D,1);
        etaSDA = thetaSDA;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PercentI = 0;
        
        for n=1:NoIter
            
            % Model:
            if (strcmp(UseAR,'yes'))
                a = ARcoeff*a + randn(D,1);
            else
                a = randn(D,1);
            end

            if (strcmp(UseNoiseInModel,'yes'))
                b = a'*Target + sqrt(NoiseVar)*randn;
            else
                b = a'*Target;
            end
            
            % Buffers required by the Prox-SVRG method.
            Buffer4a = [Buffer4a,a];
            
            DeltaOld = Delta;
            Delta = delta*Delta+1;
            DeltaSq = delta*delta*DeltaSq+1;

            R = delta*R + bsxfun(@times,a,a');
            R = (R+R')/2;
            r = delta*r + b*a;
            
            RSFM = delta*DeltaOld*RSFM/Delta + a*a'/Delta;
            RSFM = (RSFM+RSFM')/2;
            rSFM = delta*DeltaOld*rSFM/Delta + b*a/Delta;
            
            %% RLS: page 199:
            %   @Book{SayedBook,
            %   author = {Sayed, A H},
            %   title = {Adaptive Filters},
            %   publisher = {John Wiley \& Sons},
            %   year = {2008},
            %   address = {Hoboken: New~Jersey}
            % }

            AuxRLS = Prls*a;
            Prls = (1/delta)*(Prls - bsxfun(@times,AuxRLS,AuxRLS')/(delta + a'*AuxRLS));
            Prls = (Prls+Prls')/2;
            xRLS = xRLS + (b-a'*xRLS)*Prls*a;
            
            DevOptRLS(n) = norm(xRLS-Target)/TargetNorm;
            DevOptRLSav(n) = ((noexp-1)/noexp)*DevOptRLSav(n) + DevOptRLS(n)/noexp;
            
            %% l0-RLS:
            % @Article{,
            %          author = {Eksioglu, E M and Tanc, A K},
            %          title = {{RLS} algorithm with convex regularization},
            %          journal = IEEESPLett,
            %          year = {2011},
            %          volume = {18},
            %          number = {8},
            %          pages = {470--473}
            %          month = {Aug.}
            %         }

            if ( strcmp(UseConstantGammal0RLS,'yes') )
                gammal0RLS = ConstantGammal0RLS;
            else
                fl0RLS = 1 - exp(-betal0RLS*abs(xl0RLS));
                fl0RLS = sum(fl0RLS);
                epsilonPrimel0RLS = Pl0RLS*rl0RLS - xl0RLS;
                aux = Pl0RLS*Gradfl0RLS;
                gammaPrimel0RLS = 2*(trace(Pl0RLS)*(fl0RLS-rhol0RLS)/D + ...
                                     aux'*epsilonPrimel0RLS)/norm(aux)^2;
                gammal0RLS = max(0,gammaPrimel0RLS)*Multiplygammal0RLSby;
            end
            
            % Step 2.
            Auxl0RLS = Pl0RLS*a;
            kl0RLS = Auxl0RLS/(lambdal0RLS + a'*Auxl0RLS);
            % Step 3.
            ksil0RLS = b - a'*xl0RLS;
            % Step 4.
            Pl0RLS = (1/lambdal0RLS)*(Pl0RLS - bsxfun(@times,kl0RLS,a')*Pl0RLS);
            Pl0RLS = (Pl0RLS + Pl0RLS')/2;
            rl0RLS = lambdal0RLS*rl0RLS + b*a;
            % Step 5.
            xl0RLS = xl0RLS + ksil0RLS*kl0RLS - gammal0RLS*(1-lambdal0RLS)*Pl0RLS*Gradfl0RLS;
            
            Gradfl0RLS = zeros(D,1);        
            for i = 1:D
                if ( abs(xl0RLS(i)) <= 1/betal0RLS )
                    Gradfl0RLS(i) = betal0RLS*sign(xl0RLS(i)) - (betal0RLS^2)*xl0RLS(i);
                end
            end
            
            DevOptl0RLS(n) = norm(xl0RLS-Target)/TargetNorm;
            DevOptl0RLSav(n) = ((noexp-1)/noexp)*DevOptl0RLSav(n) + DevOptl0RLS(n)/noexp;

            %% l0-RLS:
            % @Article{,
            %          author = {Eksioglu, E M and Tanc, A K},
            %          title = {{RLS} algorithm with convex regularization},
            %          journal = IEEESPLett,
            %          year = {2011},
            %          volume = {18},
            %          number = {8},
            %          pages = {470--473}
            %          month = {Aug.}
            %         }

            if ( strcmp(UseConstantGammal1RLS,'yes') )
                gammal1RLS = ConstantGammal1RLS;
            else
                fl1RLS = sum(abs(xl1RLS));
                epsilonPrimel1RLS = Pl1RLS*rl1RLS - xl1RLS;
                aux = Pl1RLS*Gradfl1RLS;
                gammaPrimel1RLS = 2*(trace(Pl1RLS)*(fl1RLS-rhol0RLS)/D + ...
                                     aux'*epsilonPrimel1RLS)/norm(aux)^2;
                gammal1RLS = max(0,gammaPrimel1RLS)*Multiplygammal1RLSby;
            end
            
            % Step 2.
            Auxl1RLS = Pl1RLS*a;
            kl1RLS = Auxl1RLS/(lambdal1RLS + a'*Auxl1RLS);
            % Step 3.
            ksil1RLS = b - a'*xl1RLS;
            % Step 4.
            Pl1RLS = (1/lambdal1RLS)*(Pl1RLS - bsxfun(@times,kl1RLS,a')*Pl1RLS);
            Pl1RLS = (Pl1RLS + Pl1RLS')/2;
            rl1RLS = lambdal1RLS*rl1RLS + b*a;
            % Step 5.
            xl1RLS = xl1RLS + ksil1RLS*kl1RLS - gammal1RLS*(1-lambdal1RLS)*Pl1RLS*sign(xl1RLS);
            
            Gradfl1RLS = sign(xl1RLS);
            
            DevOptl1RLS(n) = norm(xl1RLS-Target)/TargetNorm;
            DevOptl1RLSav(n) = ((noexp-1)/noexp)*DevOptl1RLSav(n) + DevOptl1RLS(n)/noexp;

            %% OSCD-TWL:
            % @Article{,
            %          author = {Angelosante, D and Bazerque, J A and Giannakis, G B},
            %          title = {Online adaptive estimation of sparse signals: where {RLS} meets the ...
            %                   $\ell_1$-norm}, 
            %          journal = IEEETSP,
            %          year = {2010},
            %          volume = {58},
            %          number = {7},
            %          pages = {3436--3447},
            %          month = {July}
            %         }
            
            if ( Sce == 9 || Sce == 10)
                lambdaOSCD = sqrt(2*(1e-20)*log(D))*sqrt(DeltaSq);
            else
                lambdaOSCD = sqrt(2*NoiseVar*log(D))*sqrt(DeltaSq);
            end

            sPlus = sign(xOSCD);
            sMinus = -sPlus;
            
            % (20) and (21):
            dPlus = R*xOSCD - r + lambdaOSCD*sPlus;
            dMinus = r - R*xOSCD + lambdaOSCD*sMinus;

            [~,pStar] = min(min([dPlus,dMinus],[],2));
            rOSCD = r(pStar) - R(pStar,:)*xOSCD + R(pStar,pStar)*xOSCD(pStar);
            xOSCD(pStar) = sign(rOSCD)*max(0,abs(rOSCD)-lambdaOSCD)/R(pStar,pStar);
            
            DevOptOSCD(n) = norm(xOSCD-Target)/TargetNorm;
            DevOptOSCDav(n) = ((noexp-1)/noexp)*DevOptOSCDav(n) + DevOptOSCD(n)/noexp;
            
            %% OCCD-TWL.
            % @Article{,
            %          author = {Angelosante, D and Bazerque, J A and Giannakis, G B},
            %          title = {Online adaptive estimation of sparse signals: where {RLS} meets
            %          the $\ell_1$-norm}, 
            %          journal = IEEETSP,
            %          year = {2010},
            %          volume = {58},
            %          number = {7},
            %          pages = {3436--3447},
            %          month = {July}
            %         }

            for dd=1:D
                rOCCD = r(dd) - R(dd,:)*xOCCD + R(dd,dd)*xOCCD(dd);
                xOCCD(dd) = sign(rOCCD)*max(0,abs(rOCCD)-lambdaOSCD)/R(dd,dd);
            end
            
            DevOptOCCD(n) = norm(xOCCD-Target)/TargetNorm;
            DevOptOCCDav(n) = ((noexp-1)/noexp)*DevOptOCCDav(n) + DevOptOCCD(n)/noexp;
            
            %% Variance reduction method: Prox-SVRG:
            % @Article{,
            %          author = {Xiao, L and Zhang, T},
            %          title = {A proximal stochastic gradient method with progressive variance
            %          reduction}, 
            %          journal = SIAMOPT,
            %          year = {2014},
            %          volume = {24},
            %          number = {4},
            %          pages = {2057--2075}
            %         }
            
            xTildeSVRG = xSVRG;
            vTildeSVRG = RSFM*xTildeSVRG-rSFM;
            xkSVRG = xTildeSVRG;
            
            HowManyIndices = min(mSVRG,n);
            Indices = ((n-HowManyIndices+1):n);

            Buffer4xk = [];
            for k = 1:HowManyIndices
                aik = Buffer4a(:,Indices(k));
                vk = (delta^(HowManyIndices-k))*aik*(aik'*(xkSVRG- ...
                                                           xTildeSVRG))/(Delta/HowManyIndices) + ...
                     vTildeSVRG;
                aux = xkSVRG - etaSVRG*vk;
                xkSVRG = aux.*(1-(etaSVRG*WeightLossReg)./max(etaSVRG*WeightLossReg,abs(aux)));
                Buffer4xk = [Buffer4xk,xkSVRG];
            end
            xSVRG = sum(Buffer4xk,2)/HowManyIndices;

            DevOptSVRG(n) = norm(xSVRG-Target)/TargetNorm;
            DevOptSVRGav(n) = ((noexp-1)/noexp)*DevOptSVRGav(n) + DevOptSVRG(n)/noexp;
            
            %% Stochastic variance reduced gradient (SVRG)-ADMM:
            % @InProceedings{,
            %                author = {Zheng, S and Kwok, J T},
            %                title = {Fast-and-light stochastic {ADMM}},
            %                booktitle = {Proc.\ Intern.\ Joint Conf.\ Artificial Intelligence},
            %                year = {2016},
            %                pages = {2407--2413},
            %                month = {July},
            %                address = {Las~Vegas: NV: USA}
            %               }

            % Line 5.
            xkADMM = xTildeADMM;
            ykADMM = yTildeADMM;
            ukADMM = uTildeADMM;

            % Line 6.
            zTildeADMM = RSFM*xTildeADMM-rSFM;

            HowManyIndices = min(bADMM,n);
            Buffer4xkADMM = [];
            Buffer4ykADMM = [];
            % Line 7.
            for k = 1:mADMM
                
                Ik = ((n-HowManyIndices+1):n);
                
                AccumGrads = zeros(D,1);
                for l = 1:HowManyIndices
                    ai = Buffer4a(:,Ik(l));
                    AccumGrads = AccumGrads + (delta^(HowManyIndices-k))*(ai'*(xkADMM-xTildeADMM))*ai;
                end
                if (delta == 1)
                    Dividend = HowManyIndices;
                else
                    Dividend = (1-delta^HowManyIndices)/(1-delta);
                end
                GradfIk = AccumGrads/Dividend + zTildeADMM;
                
                % Line 8.
                aux = xkADMM + ukADMM;
                ykADMM = aux.*(1-(WeightLossReg/rhoADMM)./max((WeightLossReg/rhoADMM),abs(aux)));
                
                % Line 9, (11) and (12).
                xkADMM = xkADMM - etaADMM*(GradfIk + rhoADMM*(xkADMM - ykADMM + ukADMM))/ ...
                         (1+etaADMM*rhoADMM+1e-3);

                % Line 10.
                ukADMM = ukADMM + xkADMM - ykADMM;

                Buffer4xkADMM = [Buffer4xkADMM,xkADMM];
                Buffer4ykADMM = [Buffer4ykADMM,ykADMM];
            end
            
            % Line 12.
            xTildeADMM = sum(Buffer4xkADMM,2)/mADMM;
            yTildeADMM = sum(Buffer4ykADMM,2)/mADMM;
            uTildeADMM = -(1/rhoADMM)*(RSFM*xTildeADMM-rSFM);
            
            xADMM = (xTildeADMM+yTildeADMM)/2;
            
            DevOptADMM(n) = norm(xADMM-Target)/TargetNorm;
            DevOptADMMav(n) = ((noexp-1)/noexp)*DevOptADMMav(n) + DevOptADMM(n)/noexp;
            
            %% Adaptive sparse variational Bayes (ASVB)-MPL:
            % @Article{,
            %          author = {Themelis, K E and Rontogiannis, A A and Koutroumbas, K D},
            %          title = {A variational {B}ayes framework for sparse adaptive estimation},
            %          journal = IEEETSP,
            %          year = {2014},
            %          volume = {62},
            %          number = {18},
            %          pages = {4723--4736},
            %          month = {Sept.}
            %         }
            
            rASVB = lambdaASVB*rASVB + b*a;
            RASVB = lambdaASVB*RASVB + bsxfun(@times,a,a') - lambdaASVB*Aprev + A;
            RASVB = (RASVB+RASVB')/2;
            dASVB = lambdaASVB*dASVB + b^2;
            betaASVB = (D + 1/(1-lambdaASVB) + 2*rhoASVB)/(2*deltaASVB + dASVB - rASVB'*xASVB + ...
                                                           sigmaASVB'*diag(RASVB));
            for dd=1:D
                sigmaASVB(dd) = 1/betaASVB/RASVB(dd,dd);
                
                Minusdd = setdiff((1:D),dd);
                rminusi = RASVB(Minusdd,dd);
                xminusi = xASVB(Minusdd);

                xASVB(dd) = (rASVB(dd) - rminusi'*xminusi)/RASVB(dd,dd);
                
                alphaASVB(dd) = sqrt(bASVB(dd)/(betaASVB*(xASVB(dd))^2 + 1/RASVB(dd,dd)));
                gammaASVB(dd) = 1/alphaASVB(dd) + 1/bASVB(dd);
                bASVB(dd) = (1+kappaASVB)/(nuASVB + gammaASVB(dd)/2);
            end
            
            Aprev = A;
            A = diag(alphaASVB);

            DevOptASVB(n) = norm(xASVB-Target)/TargetNorm;
            DevOptASVBav(n) = ((noexp-1)/noexp)*DevOptASVBav(n) + DevOptASVB(n)/noexp;
            
            %% AC-SA:
            % @Article{,
            %          author = {Lan, G},
            %          title = {An optimal method for stochastic composite optimization},
            %          journal = MathProgramSerA,
            %          year = {2012},
            %          volume = {133},
            %          pages = {365--397}
            %         }
            
            % (33).
            betaACSA = (n+1)/2;
            gammaACSA = GammaACSA*(n+1)/2;
            
            xACSAmd = xACSA/betaACSA + (1-1/betaACSA)*xACSAag;

            % (28).
            aux = xACSA - gammaACSA*(WeightLossReg*sign(xACSAmd) + RSFM*xACSAmd-rSFM);
            xACSA = aux*TargetNorm/max(TargetNorm,norm(aux));
            
            % (29).
            xACSAag = xACSA/betaACSA + (1-1/betaACSA)*xACSAag;
            
            DevOptACSA(n) = norm(xACSAag-Target)/TargetNorm;
            DevOptACSAav(n) = ((noexp-1)/noexp)*DevOptACSAav(n) + DevOptACSA(n)/noexp;
            
            %% Stochastic dual averaging (SDA).
            % @InProceedings{,
            %                title = 	 {Stochastic composite least-squares regression with
            %                convergence rate $\mathcal{O}(1/n)$},
            %                author = 	 {Flammarion, Nicolas and Bach, Francis},
            %                booktitle = 	 {Proc.\ 2017 Conf.\ Learning Theory},
            %                pages = 	 {831--875},
            %                year = 	 {2017},
            %                volume = 	 {65},
            %                series = 	 {Proc.\ Machine Learning Research},
            %                optaddress = 	 {Amsterdam, Netherlands},
            %                month = 	 {Jul.}
            %               }
            
            etaSDA = etaSDA - gammaSDA*(RSFM*thetaSDA - rSFM);
            thetaSDA = etaSDA.*(1-(gammaSDA*lambdaSDA*n)./max(gammaSDA*lambdaSDA*n,abs(etaSDA)));
            % thetaAvSDA = (n-1)*thetaAvSDA/n + thetaSDA/n;
            
            DevOptSDA(n) = norm(thetaSDA-Target)/TargetNorm;
            DevOptSDAav(n) = ((noexp-1)/noexp)*DevOptSDAav(n) + DevOptSDA(n)/noexp;
            
            %% S-FM-HSDM.
            
            AuxResolvent = Resolvent;
            AuxResolventII = ResolventII;
            for d=1:D
                Aux = ID;
                Aux(d,:) = ID(d,:) - AuxResolvent(d,:)/(gammaI*delta*DeltaOld+AuxResolvent(d,d));
                AuxResolvent = AuxResolvent*(sparse(Aux));
                AuxResolvent = (AuxResolvent+AuxResolvent')/2;
                
                AuxII = ID;
                AuxII(d,:) = ID(d,:) - ... 
                    AuxResolventII(d,:)/(lambdaSFMII*delta*DeltaOld+AuxResolventII(d,d));
                AuxResolventII = AuxResolventII*(sparse(AuxII));
                AuxResolventII = (AuxResolventII+AuxResolventII')/2;
            end
            AuxAux = AuxResolvent*a;
            Resolvent = AuxResolvent-bsxfun(@times,AuxAux,AuxAux')/(delta*DeltaOld+AuxAux'*a);
            Resolvent = Delta*Resolvent/delta/DeltaOld;
            Resolvent = (Resolvent+Resolvent')/2;        
            AuxAuxII = AuxResolventII*a;
            ResolventII = AuxResolventII - ...
                bsxfun(@times,AuxAuxII,AuxAuxII')/(delta*DeltaOld+AuxAuxII'*a);
            ResolventII = Delta*ResolventII/delta/DeltaOld;
            ResolventII = (ResolventII+ResolventII')/2;
            
            % Version I: Adaptive constraints: T = Prox.
            TalphaMinusGrad = alphaI*TminusGrad+(1-alphaI)*IminusGrad;
            Tmap = Resolvent*(xSFMI/gammaI+rSFM);
            TminusGrad = Tmap;
            IminusGrad = xSFMI;
            MidPointSFMI = MidPointSFMI - TalphaMinusGrad + TminusGrad;
            xSFMI = MidPointSFMI.*(1-lambdaSFMI./max(lambdaSFMI,abs(MidPointSFMI)));

            DevOptSFMI(n) = norm(xSFMI-Target)/TargetNorm;
            DevOptSFMIav(n) = ((noexp-1)/noexp)*DevOptSFMIav(n) + DevOptSFMI(n)/noexp;
            
            % Version II: Standard loss.
            TalphaN = alphaII*repmat((xSFMIIprev(:,1)+xSFMIIprev(:,2))/2, [1,2]) + (1-alphaII)* ...
                      xSFMIIprev;
            TnPlusOne = repmat((xSFMII(:,1)+xSFMII(:,2))/2, [1,2]);
            xSFMIIhalf = xSFMIIhalf-TalphaN+TnPlusOne;
            xSFMIIprev = xSFMII;
            xSFMIIa = ResolventII*(xSFMIIhalf(:,1)/lambdaSFMII + rSFM);
            xSFMIIb = xSFMIIhalf(:,2).*(1 - ...
                                        WeightLossReg*lambdaSFMII./max(WeightLossReg*lambdaSFMII, ...
                                                              abs(xSFMIIhalf(:,2)))); 
            xSFMII = [xSFMIIa,xSFMIIb];
            
            DevOptSFMII(n) = norm((xSFMII(:,1)+xSFMII(:,2))/2-Target)/TargetNorm;
            DevOptSFMIIav(n) = ((noexp-1)/noexp)*DevOptSFMIIav(n) + DevOptSFMII(n)/noexp;
            
            % % Version III: Adaptive constraints: T = Proj.
            
            % TnAlphaIII = alpha*TnPrevIII + (1-alpha)*xSFMIIIprev;
            % TnPlusOneIII = xSFMIII - R*(PinvR*xSFMIII) + PinvR*r;
            % % TnPlusOneIII = xSFMIII - BasisR*(BasisR'*xSFMIII) +
            % BasisR*(PseudoEigenValues.*(BasisR'*r)); 
            % xSFMIIIhalf = xSFMIIIhalf - TnAlphaIII + TnPlusOneIII;
            % TnPrevIII = TnPlusOneIII;
            % xSFMIIIprev = xSFMIII;
            % xSFMIII = xSFMIIIhalf.*(1-lambdaSFMIII./max(lambdaSFMIII, abs(xSFMIIIhalf)));
            
            % DevOptSFMIII(n) = norm(xSFMIII-Target)/TargetNorm;
            % DevOptSFMIIIav(n) = ((noexp-1)/noexp)*DevOptSFMIIIav(n) + DevOptSFMIII(n)/noexp;
            
            % Version IV: Adaptive constraints: T = Grad.
            % rhoSFMIV = trace(RSFM);
            % rhoSFMIV = eigs(RSFM,1)+(1e-1);
            % Power iteration.
            PowerVold = PowerV;
            PowerV = RSFM*PowerV;
            PowerV = PowerV/norm(PowerV);
            rhoSFMIV = PowerV'*PowerVold/(PowerVold'*PowerVold);
            rhoSFMIV = rhoSFMIV + ConstAboveEigenV;
            
            TalphaIV = alphaIV*TIV+(1-alphaIV)*xSFMIVprev;
            xSFMIVprev = xSFMIV; 
            TIV = xSFMIV - muSFMIV*(RSFM*xSFMIV-rSFM)/rhoSFMIV;
            xSFMIVmiddle = xSFMIVmiddle-TalphaIV+TIV;
            xSFMIV = xSFMIVmiddle.*(1-lambdaSFMIV./max(lambdaSFMIV,abs(xSFMIVmiddle)));
            
            DevOptSFMIV(n) = norm(xSFMIV-Target)/TargetNorm;
            DevOptSFMIVav(n) = ((noexp-1)/noexp)*DevOptSFMIVav(n) + DevOptSFMIV(n)/noexp;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if (mod(n,NoIter/10)==0)
                PercentI = PercentI + 1;
                fprintf('.. %d%%',PercentI*10);
            end
        end
        
        FileName = strcat('l1RLSresultsScenario',int2str(Sce));
        PARAM.noexp = noexp;
        % save(FileName, 'PARAM', 'DevOptRLSav', 'DevOptOSCDav', 'DevOptOCCDav', ...
        %      'DevOptl0RLSav', 'DevOptl1RLSav', 'DevOptSVRGav', 'DevOptADMMav', ...
        %      'DevOptASVBav', 'DevOptACSAav', 'DevOptSDAav', 'DevOptSFMIIav', ...
        %      'DevOptSFMIVav', 'DevOptSFMIav');
    end

    fprintf('\n');
    PlotFigures;

end