%
%
% TSA preprocessing
%
% Sangtae Ahn (stahn@knu.ac.kr)
% Brain AI Lab.
%
% first written  5/9/2021
%
%


close all
clear
clc


%% SET PATH

dPath=['D:\OneDrive - knu.ac.kr\BrainAI\Research\Data\SSSEP\'];
eeglab;
pop_editoptions( 'option_savetwofiles', 1,'option_single', 0);

cd(dPath);
subStruct=dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'sub')));
nSub = length(subStruct);

lowCut=1;
highCut=50;

%% MAIN LOOP

for iSub = 1 : nSub
    
    subId = subStruct(iSub).name;
    fileStruct = dir([subId  '/*.set']);
    fileId = fileStruct(1).name;
    
    disp(['Sub ' num2str(iSub) ' Loading......... ' fileId]);
    EEG = pop_loadset('filepath',[dPath subId],'filename',fileId);
    
    % BPF
    disp(['band-pass filtering from '  num2str(lowCut) ' to ' num2str(highCut)  ' Hz']);
    EEG = pop_eegfiltnew(EEG, lowCut, highCut);
    
    %     [spec freq]= spectopo(EEG.data,0,EEG.srate);
    %     figure;plot(freq,spec);
    %     a=spec(:,8:25);
    %     EEG = pop_epoch( EEG, {  'left'  'right'  }, [0  3]);
    %
    %     for iTrial = 1 : 100
    %         spec(iTrial,:,:)=spectopo(EEG.data(:,:,iTrial),0,EEG.srate,'plot','off');
    %     end
    %
    %     a=spec(:,chIdx,8:25);
    %     chIdx = [4 5 6 9 10 11 12 13 14 17 18 19 39 40 41 44 45 46 49 50 51 54 55 56];
    %
    %     x_train = a([1:45 51:95],:,:);
    %     y_train = [zeros(45,1);ones(45,1)];
    %     x_test = a([46:50 96:100],:,:);
    %     y_test = [zeros(5,1);ones(5,1)];
    
    
    % ASR
    EEG.etc.historychanlocs=EEG.chanlocs;
    EEG.etc.historychaninfo=EEG.chaninfo;
    EEG = clean_rawdata(EEG,5,-1,0.8,4,5,-1); % default setting
    %     EEG.etc.badchan=find(EEG.etc.clean_channel_mask==0); %Bad chananel information from ASR
    EEG.etc.originalEEG=EEG; % keep origianl EEG before interpolation
    EEG = pop_interp(EEG, EEG.etc.historychanlocs, 'spherical');
    
    % CAR
    EEG = pop_reref( EEG, []);
    
    pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_p.set']);
    
    
    % ICA
    EEG.rank=rank(double(EEG.data));
    EEG = pop_runica(EEG,'extended',1,'pca',EEG.rank);
    pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_pi.set']);
    
    % IC Label
    EEG = pop_iclabel(EEG, 'default');
    
    rejIdx=[];
    cutProb=0.5; % 50 percent
    for iICA = 1 : EEG.rank
        [maxProb maxIdx]= max(EEG.etc.ic_classification.ICLabel.classifications(iICA, :));
        % 1: brain / 2: Muscle / 3: Eye / 4: Heart / 5: Line Noise / 6: Channel Noise / 7: Other
        if maxIdx ~= 1 && maxIdx ~= 7 && maxProb > cutProb
            rejIdx = [rejIdx iICA];
        end
    end
    
    EEG.etc.rejIdx = rejIdx;
    EEG = pop_subcomp( EEG, rejIdx, 0);
    
    pop_saveset(EEG,'filepath',[dPath subId],'filename',[fileId(1:end-4) '_pir.set']);
    
end