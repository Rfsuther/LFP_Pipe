
% 
% 
% %%
% CorrectTrials = parallel.pool.Constant(CorrectTrials);
% MissedTrials = parallel.pool.Constant(MissedTrials);


%^uncomment all to run full function
%%
% loadFromFile = 0;
% 
% if loadFromFile
% clear 
% load testFuncLoadScatter.mat
% else
% clearvars -except CorrectTrials MissedTrials
% end
lambda_pool = power(10, 1:-0.1:-9); % Define the lambd a pool




fitMetaLearner = 1;
%%    
nBags = 100 %500
nCompts =24;
firstLayMCCsCV = zeros(nBags,nCompts);
firstLayMCCFinal = zeros(nBags,nCompts);
MCCmeta = zeros(nBags,1) ;
overSampleClass = 0;  %flag set hereeeeeee
p = gcp;
for i=1:nBags %bagging loop
    TPOUT = 0; TNOUT = 0; FPOUT =0; FNOUT = 0;
    
    rng(i)
    CVCorrect = cvpartition(size(CorrectTrials.Value,1),'HoldOut',.1);
    CVMissed = cvpartition(size(MissedTrials.Value,1),'HoldOut',.1);

    if(~isempty(p))


        TrainCorrectError = [ones(size(MissedTrials.Value(training(CVCorrect),1) ,1),1);zeros(size(MissedTrials.Value(training(CVMissed),1) ,1),1)];
        TestCorrectError = [ones(size( CorrectTrials.Value(test(CVCorrect),1) ,1),1);zeros(size( MissedTrials.Value(test(CVMissed),1),1),1)];
        CVLayer1 = cvpartition(TrainCorrectError,'Kfold',5,'Stratify',true);

        storeFirstLayer = cell(1,nCompts);
        firstLayerScores = zeros(length(TrainCorrectError),nCompts);
        parfor ii=1:nCompts  %parfor

            flatTrainOutCV = cell2mat([ ...
                CorrectTrials.Value( training(CVCorrect),ii); ...datasample(,CVMissed.TrainSize)
                MissedTrials.Value( training(CVMissed),ii) ...
                ].').';
            TrainCorrectErrorCV = [[1:length(TrainCorrectError)].',TrainCorrectError];
            %start function here pass training

            storeFirstLayer{ii} = crossval(@baseLearner,flatTrainOutCV,TrainCorrectErrorCV ,'Partition',CVLayer1);
            storeFirstLayer{ii} = sortrows(vertcat(storeFirstLayer{ii}{:}));
            firstLayerScores(:,ii) = storeFirstLayer{ii}(:,2);
            firstLayMCCsCV(i,ii) = mean(storeFirstLayer{ii}(:,4)); %this line may be wrong MCC should be cal using final model first layer

        end

        flatTestOut = zeros(length(TestCorrectError),nCompts);
        if fitMetaLearner ==1
            parfor ii=1:nCompts  %parfor

                flatTrain1stLayer = cell2mat([ ...
                    CorrectTrials.Value( training(CVCorrect),ii); ...datasample(,CVMissed.TrainSize)
                    MissedTrials.Value( training(CVMissed),ii) ...
                    ].').';
                flatTest1stLayer = cell2mat([ CorrectTrials.Value(test(CVCorrect),ii ); MissedTrials.Value(test(CVMissed),ii) ].').';
                Layer1Results = cell2mat(baseLearner(flatTrain1stLayer,[zeros(length(TrainCorrectError),1),TrainCorrectError],flatTest1stLayer,[zeros(length(TestCorrectError),1),TestCorrectError]));
                firstLayMCCFinal(i,ii) = Layer1Results(1,4);
                flatTestOut(:,ii) = Layer1Results(:,2);

            end
            
            if (overSampleClass == 1)
                correctTrials = logical(TrainCorrectError);
                if(sum(correctTrials)>sum(~correctTrials)); error('true class has more sample review oversampling functions'); end
                numUpsampleTrials = sum(~correctTrials)-sum(correctTrials);
                metaTrainFlat = [datasample(firstLayerScores(correctTrials,:),numUpsampleTrials);firstLayerScores];
                metaTrainCorrectError = [ones(numUpsampleTrials,1);TrainCorrectError];
            else
                metaTrainFlat = firstLayerScores;
                metaTrainCorrectError = TrainCorrectError;
            end
            
            metaTestFlat = flatTestOut;

            % metaMDL = fitclinear(...
            %     metaTrainFlat, ...
            %     metaTrainCorrectError, ...
            %     'Learner', 'Logistic', ...
            %     'Lambda', 'auto', ...
            %     'BetaTolerance', 0.0001, ...
            %     'Cost', [0 1; 4 0] ...
            %     );
            %if slecetfeatures
            % metaMDL = fitcnb(...
            %     metaTrainFlat, ...
            %     metaTrainCorrectError, ...
            %     'Cost',  [0 1; 4 0]);
            % metaMDL = fitcsvm(...
            % metaTrainFlat, ...
            % metaTrainCorrectError, ...
            % 'KernelFunction', 'gaussian', ...
            % 'KernelScale', 'auto', ...
            % 'BoxConstraint', 1, ...
            % 'Standardize', false, ...
            % 'Cost',  [0 1; 4 0]);

            %Optimize Meta learner
            %power(10, 1:-0.1:-9)
            HyperPstruct = struct( 'Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement','MaxObjectiveEvaluations',100,'NumGridDivisions',100,...
            'ShowPlots',false,'Verbose',0,'UseParallel',true,'Repartition',true...
            ... ,'CVPartition',cvpartition(metaTrainCorrectError,'KFold',12,'Stratify',true)
            );

                
            metaMDL = fitclinear(metaTrainFlat,metaTrainCorrectError,'Learner', 'Logistic', 'Regularization','lasso', ...
                'Lambda', 'auto', 'BetaTolerance', 0.0001, 'Cost', [0 1; 4 0]); % ,'HyperparameterOptimizationOptions',HyperPstruct,'OptimizeHyperparameters',{'Lambda'}

            [Label,score] = predict(metaMDL,metaTestFlat);
            tfGroundTruth = categorical(TestCorrectError,[0 1],{'false','true'});
            tfPredict = categorical(Label,[0 1],{'false','true'});
            confusionMeta = confusionmat(tfGroundTruth,tfPredict);
            TPmeta =  confusionMeta(2,2)+.0001; TNmeta =  confusionMeta(1,1)+.0001; FPmeta =  confusionMeta(1,2)+.0001; FNmeta = confusionMeta(2,1)+.0001;
            MCCmeta(i) = (TPmeta.*TNmeta-FPmeta.*FNmeta)./sqrt((TPmeta+FPmeta).*(TPmeta+FNmeta).*(TNmeta+FPmeta).*(TNmeta+FNmeta));
        end

        %older loop
    else
        error('parfor didnt work')
    end
end
beep
display(strcat("Meta learner oversample = ",string(logical(overSampleClass)), " MCC = ",string(mean(MCCmeta,'omitmissing'))," plusminus ",string(std(MCCmeta,'omitmissing'))))

function storeFirstLayer = baseLearner(Xtrain,Ytrain,Xtest,Ytest)
    
    falsePermute = false;

    if falsePermute;    Ytrain = Ytrain(randperm(length(Ytrain)),:);  end



    overSampleClass = 0;
    
    if (overSampleClass == 1)
        correctTrials = logical(Ytrain(:,2));
        if(sum(correctTrials)>sum(~correctTrials)); error('true class has more sample review oversampling functions'); end
        numUpsampleTrials = sum(~correctTrials)-sum(correctTrials);
        flatTrainIn = [datasample(Xtrain(correctTrials,:),numUpsampleTrials);Xtrain];
        TrainCorrectError = [ones(numUpsampleTrials,1);Ytrain(:,2)];
    else
        flatTrainIn = Xtrain;
        TrainCorrectError = Ytrain(:,2);
    end
    % MDL = fitcsvm(...
        % flatTrainIn, TrainCorrectError, 'KernelFunction', 'polynomial', 'Solver','L1QP', ...
        % 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 9, ...
        % 'Standardize', true, 'Cost', [0 1; 4 0], 'ClassNames', [0; 1]);  %  ,,,'Prior','uniform'
        MDL =   fitcknn( ...
                flatTrainIn, ...
            TrainCorrectError, ...
            "Distance","Cosine",'NumNeighbors',58,"DistanceWeight","equal",...
            'Standardize',true,...
            'Cost',[0 1; 4 0]);


    [Label,score] = predict(MDL,Xtest);
    tfGroundTruth = categorical(Ytest(:,2),[0 1],{'false','true'});
    tfPredict = categorical(Label,[0 1],{'false','true'});
    cm = confusionmat(tfGroundTruth,tfPredict);
    TPIN =  cm(2,2)+.0001;
    TNIN =  cm(1,1)+.0001;
    FPIN =  cm(1,2)+.0001;
    FNIN = cm(2,1)+.0001;
    MCCIN = (TPIN.*TNIN-FPIN.*FNIN)./sqrt((TPIN+FPIN).*(TPIN+FNIN).*(TNIN+FPIN).*(TNIN+FNIN));
    storeFirstLayer = {[Ytest(:,1),score(:,1),Xtest(:,1),(ones(length(Ytest),1)*MCCIN)]};
end
