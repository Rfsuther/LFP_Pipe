%output data with wavletscattering which useses an InvarianceScale

function   totalTrials = OutputWaveletScattering( FTDataStruct, tWindow, FolderName,InvarianceCoeff)
    removedData = FTDataStruct ;
    downSampleBy = 2;
    A_3dTrialVect = permute([cat(3,removedData.trial{:})] ,[2,1,3]);
    A_3dTrialVectShort = resample(A_3dTrialVect,1,downSampleBy); %downsamples
    CorrectOrError = logical(removedData.trialinfo(:,1));
    fsample = removedData.fsample;
    waveFltrBank = waveletScattering('SignalLength',size(A_3dTrialVectShort,1), ...
        'SamplingFrequency',fsample/downSampleBy,'InvarianceScale',InvarianceCoeff); %.75 is best
    
    [waveFeatures,U] = featureMatrix(waveFltrBank,A_3dTrialVectShort);
    waveFeatures1 = reshape(waveFeatures,[],size(waveFeatures,3),size(waveFeatures,4));


    %[A,B] = scatteringTransform(waveFltrBank,squeeze (A_3dTrialVectShort(:,1,1)));
   % scattergram(A,B,'FilterBank',1)

%%

    %visualize features
    % waveFeaturesMat = mean(waveFeatures(:,:,:,CorrectOrError),4)-mean(waveFeatures(:,:,:,~CorrectOrError),4);
    % for i =  1:size(waveFeaturesMat,3)
    %     nexttile;
    %     imagesc(waveFeaturesMat(:,:,i))
    % end
    numTrls = size(A_3dTrialVect,3);
    numChans = size(A_3dTrialVect,2);
    waveFeatures3 = cell(numTrls,numChans);
    for j = 1:numChans
        for jj = 1:numTrls
            waveFeatures3{jj,j} = waveFeatures1(:,j,jj);
        end
    end
    
    CorrectTrials = parallel.pool.Constant(waveFeatures3(logical(CorrectOrError),:,:));
    MissedTrials = parallel.pool.Constant(waveFeatures3(~logical(CorrectOrError),:,:));
    totalTrials = {CorrectTrials,MissedTrials};
end  
