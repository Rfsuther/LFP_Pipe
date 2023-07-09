%% This script is used for reading raw data from LFP datasets
% Required toolboxes: Statistics and Machine Learning Toolbox, \
% Wavlet Toolbox,  Signal Processing Toolbox, FieldTrip
% Author: Robert Sutherland
% Project: Multi-scale Memory Decoding Model

addpath('C:\Users\Rob\Desktop\MATLAB\fieldtrip\'); %%windows;
ft_defaults

%{
	Please provide the raw data in the following format

Input raw un-epoched data with event markers at time points within trials. 

1. CorrectOrError| Type: numeric or logical, Description: A column vector that has correct trials. 1/true= correct 0/false=error

2. Sample_Resp| Type: numeric, Description: A vector with each event marker as a numerical entry in time points. I.E if you have an event at 10s in dataset with 512 sample rate then event is at 5121. The number of events will equal the number of created trials.

3. dataStructRaw| Type: struct
Description: A struct wth field that house trial data as follows

3a.		trial|  Type:Cell Description: A cell containing all raw data in the format  channels x timepoints assumes data are in uV

3b. 	time|  Type:Cell Description: A cell containing all timepoints in a row vector

3b. 	label|  Type:Cell Description: A cell array with n channels containing all the lables of the channel can be a nuemerical index or charcter string

3c.		DataSetName|Type:String or char-array Description: The name of the dataset

3d.		windowLen| Type: numeric, Description: A number in seconds of the total window length centered at event marker. NOTE: The window length centerd on any event marker should not contain points outside the trial. I.E if first trial starts at t=5s then the window should not exceed 10s.
%}

% Open raw data file

%NOTE: Please write your own datafile formatting script. The only requirments are that it outputs the data in the format
% described above
clear; close all; clc;
LoadData_Kaha; 

%Controls: use these if statements to modify pipeline
intermediatePlots = false; % flag to control plots showing data progress
VisualInspection = false; 



numChannel = size(dataStructRaw.trial{1},1);
numPoints = size(dataStructRaw.trial{1},2);
windowLen = dataStructRaw.windowLen; 
DataSetName= dataStructRaw.DataSetName;
dataStructRaw = rmfield(dataStructRaw,'DataSetName');
numTrial  = length(Sample_Resp);
disp(['This dataset contains ',mat2str(numChannel),' channels and ', mat2str(numPoints),' recording timestamps'])

%% Begin FieldTrip Preprocessing
%Create a FieldTrip data struct to begin preprocessing

trialLength = windowLen * dataStructRaw.fsample; 

cfg = {};
cfg.trl = zeros(numTrial,3);
for i=1:numTrial
    trialStarts = floor(Sample_Resp(i))-trialLength/2;
    trialEnds =  floor(Sample_Resp(i))+trialLength/2-1;
    trialTrig = Sample_Resp(i);
    cfg.trl(i,:) = [trialStarts, trialEnds, trialTrig];
end

cfg.hdr = {};
cfg.hdr.chantype = 'unknown';
cfg.hdr.chanunit = 'uV';



%Preprocessing Step 1: Remove line-noise and harmonics using a frequncy domain interpolation around 60 and harmonics 
%NOTE assumes 60Hz line noise. If you have 50Hz please update this directly
if (intermediatePlots) %saves one trial from plotting if nessecary 
    trial1TimeIndex = [-windowLen/2,windowLen/2]*dataStructRaw.fsample+Sample_Resp(1);
    oneRawTrial = dataStructRaw.trial{1}(1,trial1TimeIndex(1):trial1TimeIndex(2)-1); 
end

dataStructRaw.trial{1} = ft_preproc_dftfilter(dataStructRaw.trial{1}, dataStructRaw.fsample, [60,120,180],'dftreplace','neighbour','dftwidth',3);



%Epoch 
dataRawSeged = ft_redefinetrial(cfg, dataStructRaw);
clearvars dataStructRaw;
clc
if (intermediatePlots) %can plot data to see progression of pipeline
    figure;
    subplot(3,2,1);
    plot(linspace(-windowLen/2,windowLen/2,windowLen*dataRawSeged.fsample),oneRawTrial);
    xlabel('Time (s)');
    ylabel('Voltage (uV)');
    title('raw signal');
    subplot(3,2,2)
    [pSpectTrl,freqsAxis] = pspectrum(oneRawTrial,dataRawSeged.fsample);
    freqMax = find(freqsAxis>250,1);
    plot(freqsAxis(1:freqMax),log(pSpectTrl(1:freqMax)));
    xlabel('Freq (Hz)');
    ylabel('Log Magnitude');
    title('raw signal power spectrum');

    subplot(3,2,3);
    plot(linspace(-windowLen/2,windowLen/2,windowLen*dataRawSeged.fsample),dataRawSeged.trial{1}(1,:));
    xlabel('Time (s)');
    ylabel('Voltage (uV)');
    title('removed line noise');
    subplot(3,2,4)
    [pSpectTrl,freqsAxis] = pspectrum(dataRawSeged.trial{1}(1,:),dataRawSeged.fsample);
    plot(freqsAxis(1:freqMax),log(pSpectTrl(1:freqMax)));
    xlabel('Freq (Hz)');
    ylabel('Log Magnitude');
    title('removed line noise power spectrum');
end

%Perform a global Bandpass filter to isolate LFP singal
%6th order low and high pass between 2 and 200 Hz 
cfg.lpfilter = 'yes';
cfg.lpfiltord = 6;
cfg.hpfilter = 'yes';
cfg.hpfiltord = 2;
cfg.lpfreq = 150;
cfg.hpfreq = 1;
%cfg.trialdef.prestim = -windowLen/2;
%cfg.trialdef.poststim = windowLen/2;


dataDenoised = ft_preprocessing(cfg,dataRawSeged);
dataDenoised.trialinfo =[CorrectOrError ,[1:length(CorrectOrError )].']; %added trial infofield to store binnaryLabel
dataDenoised.dataStamp = clock; %timestamp for archiving purpose
%
if(intermediatePlots)
    subplot(3,2,5);
    plot(linspace(-windowLen/2,windowLen/2,windowLen*dataDenoised.fsample),dataDenoised.trial{1}(1,:));
    xlabel('Time (s)');
    ylabel('Voltage (uV)');
    title('removed line noise and filtered');
    subplot(3,2,6)
    [pSpectTrl,freqsAxis] = pspectrum(dataDenoised.trial{1}(1,:),dataDenoised.fsample);
    plot(freqsAxis(1:freqMax),log(pSpectTrl(1:freqMax)));
    xlabel('Freq (Hz)');
    ylabel('Log Magnitude');
    title('removed line noise filtered power spectrum');
end
if (intermediatePlots)
    ft_databrowser([],dataRawSeged)
end
clear vars dataRawSeged;



%% Washing Machine
 %dataDenoised = load('dataDwnsmp').dataDwnsmp;
 
if VisualInspection
    % dataIn.datatype = 'raw';
    %{ 
    Preform Visual Artifact Rejection
    %The user will need to examine the dataset and visually select data that
    %has been coruptead the function will remove the entire channel for that
    %trial later the channel will be filled with the average of all channels
    %for that trial that did not have an artifact to keep the formating
    %consistant
    
    %inputs
    %strcut cfg: list of ...
    %struct data: ...
    %outputs
    %struct cleanData  : ...
    % added field (removed chans)
    %}
    removedData = ft_rejectvisual([], dataDenoised);
else
    removedData = dataDenoised;
end
% removedData.trialinfo = [1 0	1	1	1	1	1	1	0	0	1	0	1	0	1	0	1	0	0	0	1	1	1	1	1	0	0	0	0	0	0	0	0	1	0	1	0	0	1	1	0	1	0	0	0	1	0	1	0	0	0	0	1	1	1	1	0	0	1	1	1	0	0	1	1	1	0	0	0	0	0	0	0	0	1	0	0	0	1	1	0	0	1	1	1	0	1	0	0	0	1	1	1	1	1	1	1	1	0	1	0	0	0	1	0	0	1	0	0	0	0	0	1	0	1	1	0	1	0	0	1	1	0	0	0	0	1	0	1	0	0	1	0	0	0	1	0	1	0	1];
% removedData.trialinfo = [removedData.trialinfo',removedData.cfg.trials'];
clear dataDenoised dataRawSaved
%% DIM REDUCTION


usePCA = false;
useICA = false;
% 
if(usePCA || useICA)
    BigMAT = [];
    for i = 1:size(removedData.trial,2)
        BigMAT = [BigMAT,removedData.trial{i}];
    end
    BigMAT = BigMAT.';
    % trainingRaw= BigMAT(training1,:);  %TODO use only training data 
    % testRaw = BigMAT(testing1,:);
    if(usePCA&&~useICA)
        [coeff,score,~,~,explained,mu] = pca(BigMAT);  
        keepNum = 20;
        for i = 1:size(removedData.trial,2)
            removedData.trial{i} = [(removedData.trial{i}.'-mu)*coeff(:,1:20)].'; %ICAtransfom.transform(removedData.trial{i}.')';%
        end
    elseif(~usePCA&&useICA)
    ICAtransfom = rica(BigMAT,10);
    for i = 1:size(removedData.trial,2)
        removedData.trial{i} =ICAtransfom.transform(removedData.trial{i}.')';% [[removedData.trial{i}.'-mu]*coeff(:,1:20)].'; %
    end
    keepNum = 10;


    makeLabel = [];
    for i=1:keepNum
        makeLabel  = [makeLabel,{sprintf('PC%d',i)}];
    end
    removedData.label  = makeLabel.';
    else
        error('Please only use PCA OR ICA')
    end
end


%clip trials
trlLen = length(removedData.trial{1}(1,:));
for i=2:length(removedData.trial)
    trlLen = min(trlLen,length(removedData.trial{i}(1,:)));
end

trlLen = length(removedData.trial{i}(1,:));
for i=2:length(removedData.trial)
    removedData.trial{i} = removedData.trial{i}(:,1:trlLen);
end
%%
for i = 1:15

InvarianceCoeffMAT = 1.15.^[1:15]-1;

Tfreq_Features = OutputWaveletScattering(removedData , windowLen, DataSetName,InvarianceCoeffMAT(i));

%
CorrectTrials = Tfreq_Features{1};
MissedTrials = Tfreq_Features{2};
display(InvarianceCoeffMAT(i));
metaClassifierStatsSCATTERING;
end