

%
%
% TSA data transform to EEGLAB
%
% Sangtae Ahn (stahn@knu.ac.kr)
% Brain AI Lab.
%
% first written by 05/13/2021
%
%



close all
clear
clc


%% Load dataset
MainPath = ['D:\OneDrive - University of North Carolina at Chapel Hill\MATLAB\'];

addpath([MainPath 'toolbox\eeglab2019_1']);
addpath([MainPath 'SSSEP']);

eeglab;
pop_editoptions( 'option_savetwofiles', 1,'option_single', 0);

dPath=[MainPath 'SSSEP\Data'];
cd(dPath);

subStruct=dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'.set')));
nSub = length(subStruct);

%%

for iSub = 1 : nSub
    eeglab
    load(['sub' num2str(iSub) '_SSSEP']);
    
    data=[eeg.raw_left eeg.raw_right];
    EEG.data = data./32; % gain 32 
    EEG.srate = eeg.srate;
    nTrial = eeg.n_trials;
    
    % LEFT
    idxLeft = find(eeg.event(1:size(EEG.data,2)/2)==1);
    for iIdx = 1 : length(idxLeft)
        EEG.event(iIdx).latency=  idxLeft (iIdx);
        EEG.event(iIdx).type=  'left';
    end
    
    % RIGHT
    idxRight = idxLeft + size(EEG.data,2)/2;
    for iIdx = length(idxRight)+1 : length(idxRight)*2
        EEG.event(iIdx).latency=  idxRight (iIdx-nTrial);
        EEG.event(iIdx).type=  'right';
    end
    
    EEG.chanlocs = readlocs('biosemi64.ced');
    eeglab redraw;
    
    pop_saveset(EEG,'filepath',dPath,'filename',['SSSEP_sub' num2str(iSub) '.set']);
    
    

end
    





