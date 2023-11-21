%{
Time series to be computed
IHR
PTT
EDAPh
EDATon
RESP

Other signals not used in the analysis:
TEMP
SEMGEnv
ZEMGEnv
CEMGEnv
%}

close all
clear all
addpath(".\utils\")

DataFolder = '.\data'; % path to the dataset

folders = {...
    'scenario_1\',...
    'scenario_2\fold_0\',...
    'scenario_2\fold_1\',...
    'scenario_2\fold_2\',...
    'scenario_2\fold_3\',...
    'scenario_2\fold_4\',...
    'scenario_3\fold_0\',...
    'scenario_3\fold_1\',...
    'scenario_3\fold_2\',...
    'scenario_3\fold_3\',...
    'scenario_4\fold_0\',...
    'scenario_4\fold_1\',...
};
N_folders = length(folders);
for n_folder = 1 : N_folders
    FsPhy = 1000; FsAnn = 20;
    mkdir([DataFolder,folders{n_folder},'train\preprocessed'])
    outfolder = [DataFolder,folders{n_folder},'train\preprocessed\'];
    FilesPhy = dir([DataFolder,folders{n_folder},'train\physiology\*.csv']);
    disp(['Working on folder ',folders{n_folder}])
    for n_file = 1 : length(FilesPhy)
        data = readtable([FilesPhy(n_file).folder,'\',FilesPhy(n_file).name]);
        disp(['File ',FilesPhy(n_file).name,' train'])
        FileAnno = dir([DataFolder,folders{n_folder},'train\annotations\',...
            FilesPhy(n_file).name]);
        annotations = readtable([FileAnno(1).folder,'\',FileAnno(1).name]);

        %% ECG
        disp('ECG')
        [QRSval,QRSi,~]=pan_tompkin(data.ecg,FsPhy,0);
        QRStime = data.time(QRSi+13)/1000;
        % figure; plot(data.time/1000,data.ecg); hold on; plot(QRStime,data.ecg(QRSi+13),'rx')
        RRsegments = diff(QRStime);
        RRsegTimes = QRStime(2:end);
        hri = 60./RRsegments; % bpm
        IHR = spline(RRsegTimes, hri, annotations.time/1000);
        % figure; plot(RRsegTimes, hri); hold on; plot(annotations.time/1000,HR)
        % figure; plot(annotations.time/1000,HR)

        %% PPG
        disp('PPG')
        [b,a] = butter(4,([0.5 16]/(FsPhy/2)));
        PPG_filtered = filtfilt(b,a,data.bvp);
        [~, PPGpeaksi200, ~, ~, time, wave] = BP_annotate(PPG_filtered, FsPhy, 'false');
        PPGpeaksi = PPGpeaksi200*5;
        % figure; plot(data.time,data.bvp); hold on; plot(data.time(PPGpeaksi),data.bvp(PPGpeaksi),'rx')
        PPGpeakTime = data.time(PPGpeaksi)/1000;
        PPGpeakTime = unique(PPGpeakTime);

        PTTraw = [];
        PTTtime = [];
        for npeak = 1 : length(PPGpeakTime)
            thisQRSpeak = QRStime(find(QRStime<PPGpeakTime(npeak),1,'last'));
            if ~isempty(thisQRSpeak) && (PPGpeakTime(npeak) - thisQRSpeak < 1)
                PTTraw = [PTTraw;(PPGpeakTime(npeak) - thisQRSpeak)];
                PTTtime = [PTTtime;PPGpeakTime(npeak)];
            end
        end
        PTT = spline(PTTtime, PTTraw, annotations.time/1000);

        %{
        figure 
        subplot(2,1,1)
        yyaxis left
        plot(data.time/1000,data.bvp); hold on; 
        plot(PPGpeakTime,data.bvp(PPGpeaksi),'rx')
        yyaxis right
        plot(data.time/1000,data.ecg); hold on
        plot(QRStime,data.ecg(QRSi+13),'rx')
        subplot(2,1,2)
        plot(PTTtime, PTTraw); hold on; plot(annotations.time/1000,PTT)
        %}

        %% EMG
        disp('EMG')
        Fn = [15 400];
        [b_bp,a_bp] = butter(4,Fn/(FsPhy/2));
        emg_trap = filtfilt(b_bp,a_bp,data.emg_trap);
        emg_zygo = filtfilt(b_bp,a_bp,data.emg_zygo);
        emg_coru = filtfilt(b_bp,a_bp,data.emg_coru); 

        %{
        resolutonReqd = 0.01; NFFT = FsPhy / resolutonReqd; window = 256; overlap = 128;
        [Pemg_trap,f] = pwelch(emg_trap-mean(emg_trap),window,overlap,NFFT,FsPhy); 
        [Pemg_zygo,f] = pwelch(emg_zygo-mean(emg_zygo),window,overlap,NFFT,FsPhy); 
        [Pemg_coru,f] = pwelch(emg_coru-mean(emg_coru),window,overlap,NFFT,FsPhy);
        figure; 
        subplot(3,1,1); plot(f,Pemg_trap)
        subplot(3,1,2); plot(f,Pemg_zygo)
        subplot(3,1,3); plot(f,Pemg_coru)
        %}

        TEMGEnv = decimate(movmean(abs(emg_trap),FsPhy/100,'Endpoints','shrink'),50); 
        ZEMGEnv = decimate(movmean(abs(emg_zygo),FsPhy/100,'Endpoints','shrink'),50); 
        CEMGEnv = decimate(movmean(abs(emg_coru),FsPhy/100,'Endpoints','shrink'),50); 

        %{
        figure
        subplot(3,1,1); plot(data.time/1000,emg_trap); hold on; plot(annotations.time/1000,TEMGEnv)
        subplot(3,1,2); plot(data.time/1000,emg_zygo); hold on; plot(annotations.time/1000,ZEMGEnv)
        subplot(3,1,3); plot(data.time/1000,emg_coru); hold on; plot(annotations.time/1000,CEMGEnv)
        %}

        %% EDA
        disp('EDA')
        EDA = decimate(data.gsr,50);
        EDAfilt = medfilt1(EDA,FsAnn);
        EDAfilt(1) = EDAfilt(2);
        bhi = fir1(34,1/(FsAnn/2));
        EDAfir = filtfilt(bhi,1,EDAfilt);

        % figure; plot(data.time/1000,data.gsr); hold on; plot(annotations.time/1000,EDAfir); legend('Raw EDA','Filtered EDA')

        meanEDA = mean(EDAfir); stdEDA = std(EDAfir);
        [phasic, phasic_sparse, tonic, ~, ~, ~, ~] = cvxEDA(zscore(EDAfir), 1/FsAnn);
        EDAPh = phasic*stdEDA;
        EDATon = tonic*stdEDA + meanEDA;
        %{
        figure
        yyaxis left
        plot(data.time/1000,data.gsr); hold on
        plot(annotations.time/1000,EDATon,'m')
        yyaxis right
        plot(annotations.time/1000,EDAPh)    
        legend('Raw EDA','Tonic EDA','Phasic EDA')
        %}

        %% RESPIRATION
        disp('RESP')
        Factor = 250;
        filteredRESP = decimate(data.rsp-mean(data.rsp),Factor);
        FsRESP = FsPhy/Factor;
        timeRESP = (0:length(filteredRESP)-1)/FsRESP;
        [~,RESPpeaksLocs] = findpeaks(filteredRESP);
        %{
        figure; 
        plot(data.time/1000,data.rsp-mean(data.rsp)); hold on; 
        plot(timeRESP,filteredRESP)
        plot(timeRESP(RESPpeaksLocs),filteredRESP(RESPpeaksLocs),'rx')
        %}
        RespRate = 60./diff(timeRESP(RESPpeaksLocs));
        RespRateTimes = timeRESP(RESPpeaksLocs(2:end));
        RESP = spline(RespRateTimes, RespRate, annotations.time/1000);
        % figure; plot(annotations.time/1000, RESP)
        %% TEMPERATURE
        disp('TEMP')
        TEMP = decimate(data.skt,50);
        % figure; plot(data.time/1000,data.skt); hold on; plot(annotations.time/1000,TEMP)
        
        Output = table(annotations.time,IHR,PTT,EDAPh,EDATon,RESP,TEMP,...
            TEMGEnv,ZEMGEnv,CEMGEnv,annotations.valence,annotations.arousal);
        Output.Properties.VariableNames = {'time','IHR','PTT','EDAPh',...
            'EDATon','RESP','TEMP','TEMGEnv','ZEMGEnv','CEMGEnv',...
            'Valence','Arousal'};
        disp('Creating table and saving...')
        writetable(Output,[outfolder,FilesPhy(n_file).name]);
        close all
    end
    
    %% Test
    mkdir([DataFolder,folders{n_folder},'test\preprocessed'])
    outfolder = [DataFolder,folders{n_folder},'test\preprocessed\'];
    FilesPhy = dir([DataFolder,folders{n_folder},'test\physiology\*.csv']);
    for n_file = 1 : length(FilesPhy)
        disp(['File ',FilesPhy(n_file).name,' test'])
        data = readtable([FilesPhy(n_file).folder,'\',FilesPhy(n_file).name]);   
        timeAnnotations = 1000*(0:1/FsAnn:max(data.time/1000))';
        annotations = table(timeAnnotations,zeros(numel(timeAnnotations),1),zeros(numel(timeAnnotations),1));
        annotations.Properties.VariableNames = {'time','valence','arousal'};
        
        %% ECG
        disp('ECG')
        [QRSval,QRSi,~]=pan_tompkin(data.ecg,FsPhy,0);
        QRStime = data.time(QRSi+13)/1000;
        % figure; plot(data.time/1000,data.ecg); hold on; plot(QRStime,data.ecg(QRSi+13),'rx')
        RRsegments = diff(QRStime);
        RRsegTimes = QRStime(2:end);
        hri = 60./RRsegments; % bpm
        IHR = spline(RRsegTimes, hri, annotations.time/1000);
        % figure; plot(RRsegTimes, hri); hold on; plot(annotations.time/1000,IHR)
        % figure; plot(annotations.time/1000,IHR)

        %% PPG
        disp('PPG')
        [b,a] = butter(4,([0.5 16]/(FsPhy/2)));
        PPG_filtered = filtfilt(b,a,data.bvp);
        [~, PPGpeaksi200, ~, ~, time, wave] = BP_annotate(PPG_filtered, FsPhy, 'false');
        PPGpeaksi = PPGpeaksi200*5;
        % figure; plot(data.time,data.bvp); hold on; plot(data.time(PPGpeaksi),data.bvp(PPGpeaksi),'rx')
        PPGpeakTime = data.time(PPGpeaksi)/1000;
        PPGpeakTime = unique(PPGpeakTime);

        PTTraw = [];
        PTTtime = [];
        for npeak = 1 : length(PPGpeakTime)
            thisQRSpeak = QRStime(find(QRStime<PPGpeakTime(npeak),1,'last'));
            if ~isempty(thisQRSpeak) && (PPGpeakTime(npeak) - thisQRSpeak < 1)
                PTTraw = [PTTraw;(PPGpeakTime(npeak) - thisQRSpeak)];
                PTTtime = [PTTtime;PPGpeakTime(npeak)];
            end
        end
        PTT = spline(PTTtime, PTTraw, annotations.time/1000);

        %{
        figure 
        subplot(2,1,1)
        yyaxis left
        plot(data.time/1000,data.bvp); hold on; 
        plot(PPGpeakTime,data.bvp(PPGpeaksi),'rx')
        yyaxis right
        plot(data.time/1000,data.ecg); hold on
        plot(QRStime,data.ecg(QRSi+13),'rx')
        subplot(2,1,2)
        plot(PTTtime, PTTraw); hold on; plot(annotations.time/1000,PTT)
        %}

        %% EMG
        disp('EMG')
        Fn = [15 400];
        [b_bp,a_bp] = butter(4,Fn/(FsPhy/2));
        emg_trap = filtfilt(b_bp,a_bp,data.emg_trap);
        emg_zygo = filtfilt(b_bp,a_bp,data.emg_zygo);
        emg_coru = filtfilt(b_bp,a_bp,data.emg_coru); 

        %{
        resolutonReqd = 0.01; NFFT = FsPhy / resolutonReqd; window = 256; overlap = 128;
        [Pemg_trap,f] = pwelch(emg_trap-mean(emg_trap),window,overlap,NFFT,FsPhy); 
        [Pemg_zygo,f] = pwelch(emg_zygo-mean(emg_zygo),window,overlap,NFFT,FsPhy); 
        [Pemg_coru,f] = pwelch(emg_coru-mean(emg_coru),window,overlap,NFFT,FsPhy);
        figure; 
        subplot(3,1,1); plot(f,Pemg_trap)
        subplot(3,1,2); plot(f,Pemg_zygo)
        subplot(3,1,3); plot(f,Pemg_coru)
        %}

        TEMGEnv = decimate(movmean(abs(emg_trap),FsPhy/100,'Endpoints','shrink'),50); 
        ZEMGEnv = decimate(movmean(abs(emg_zygo),FsPhy/100,'Endpoints','shrink'),50); 
        CEMGEnv = decimate(movmean(abs(emg_coru),FsPhy/100,'Endpoints','shrink'),50); 

        %{
        figure
        subplot(3,1,1); plot(data.time/1000,emg_trap); hold on; plot(annotations.time/1000,TEMGEnv)
        subplot(3,1,2); plot(data.time/1000,emg_zygo); hold on; plot(annotations.time/1000,ZEMGEnv)
        subplot(3,1,3); plot(data.time/1000,emg_coru); hold on; plot(annotations.time/1000,CEMGEnv)
        %}

        %% EDA
        disp('EDA')
        EDA = decimate(data.gsr,50);
        EDAfilt = medfilt1(EDA,FsAnn);
        EDAfilt(1) = EDAfilt(2);
        bhi = fir1(34,1/(FsAnn/2));
        EDAfir = filtfilt(bhi,1,EDAfilt);

        % figure; plot(data.time/1000,data.gsr); hold on; plot(annotations.time/1000,EDAfir); legend('Raw EDA','Filtered EDA')

        meanEDA = mean(EDAfir); stdEDA = std(EDAfir);
        [phasic, phasic_sparse, tonic, ~, ~, ~, ~] = cvxEDA(zscore(EDAfir), 1/FsAnn);
        EDAPh = phasic*stdEDA;
        EDATon = tonic*stdEDA + meanEDA;
        %{
        figure
        yyaxis left
        plot(data.time/1000,data.gsr); hold on
        plot(annotations.time/1000,EDATon,'m')
        yyaxis right
        plot(annotations.time/1000,EDAPh)    
        legend('Raw EDA','Tonic EDA','Phasic EDA')
        %}

        %% RESPIRATION
        disp('RESP')
        Factor = 250;
        filteredRESP = decimate(data.rsp-mean(data.rsp),Factor);
        FsRESP = FsPhy/Factor;
        timeRESP = (0:length(filteredRESP)-1)/FsRESP;
        [~,RESPpeaksLocs] = findpeaks(filteredRESP);
        %{
        figure; 
        plot(data.time/1000,data.rsp-mean(data.rsp)); hold on; 
        plot(timeRESP,filteredRESP)
        plot(timeRESP(RESPpeaksLocs),filteredRESP(RESPpeaksLocs),'rx')
        %}
        RespRate = 60./diff(timeRESP(RESPpeaksLocs));
        RespRateTimes = timeRESP(RESPpeaksLocs(2:end));
        RESP = spline(RespRateTimes, RespRate, annotations.time/1000);
        % figure; plot(annotations.time/1000, RESP)
        %% TEMPERATURE
        disp('TEMP')
        TEMP = decimate(data.skt,50);
        % figure; plot(data.time/1000,data.skt); hold on; plot(annotations.time/1000,TEMP)
        
        Output = table(annotations.time,IHR,PTT,EDAPh,EDATon,RESP,TEMP,...
            TEMGEnv,ZEMGEnv,CEMGEnv,annotations.valence,annotations.arousal);
        Output.Properties.VariableNames = {'time','IHR','PTT','EDAPh',...
            'EDATon','RESP','TEMP','TEMGEnv','ZEMGEnv','CEMGEnv',...
            'Valence','Arousal'};
        disp('Creating table and saving data...')
        writetable(Output,[outfolder,FilesPhy(n_file).name]);
        close all
    end
    disp('Next folder!')
end
  