classdef AudioPredictionApp < matlab.apps.AppBase
    
    properties (Access = public)
        % Core UI
        UIFigure          matlab.ui.Figure
        RootGrid          matlab.ui.container.GridLayout
        TitleLabel        matlab.ui.control.Label
        
        % Button row
        ButtonGrid        matlab.ui.container.GridLayout
        UploadButton      matlab.ui.control.Button
        PlayButton        matlab.ui.control.Button
        RecordButton      matlab.ui.control.Button
        StopButton        matlab.ui.control.Button
        UseRecordingButton matlab.ui.control.Button
        
        % Results card
        ResultsPanel      matlab.ui.container.Panel
        ResultsGrid       matlab.ui.container.GridLayout
        GenderLabel       matlab.ui.control.Label
        AgeLabel          matlab.ui.control.Label
        
        % Status bar
        StatusLabel       matlab.ui.control.Label
        
        % Audio state
        audioData         double
        sampleRate        double
        audioFilename     char
        
        % Recorder
        recorder          audiorecorder
        isRecording       logical = false
        lastRecordingPath char = ''   % saved .wav of last mic recording
    end
    
    methods (Access = private)
        
        function startupFcn(app)
            app.StatusLabel.Text = 'Ready. Upload a file or use the mic.';
        end
        
        % ===== File upload flow =====
        function UploadButtonPushed(app, ~)
            [file, path] = uigetfile({'*.wav;*.WAV;*.mp3', 'Audio Files (*.wav, *.mp3)'}, 'Select Audio File');
            if isequal(file,0) || isequal(path,0)
                app.StatusLabel.Text = 'No file selected.';
                return;
            end
            app.audioFilename = fullfile(path, file);
            app.StatusLabel.Text = 'Loading audio...';
            
            % Clear previous audio data
            app.audioData = [];
            app.sampleRate = [];
            app.PlayButton.Enable = 'off';
            
            try
                [audioTemp, fsTemp] = audioread(app.audioFilename);
                if isempty(audioTemp), error('Audio is empty.'); end
                if size(audioTemp,2) > 1
                    audioTemp = mean(audioTemp,2);
                end
                maxVal = max(abs(audioTemp));
                if maxVal > 0
                    audioTemp = audioTemp / maxVal;
                else
                    error('Audio contains only silence');
                end
                app.audioData = audioTemp;
                app.sampleRate = fsTemp;
                app.StatusLabel.Text = sprintf('Audio loaded: %s (%.1fs, %d Hz)', file, numel(audioTemp)/fsTemp, fsTemp);
                app.PlayButton.Enable = 'on';
                
                app.processAudio();  % run predictions
            catch ME
                app.StatusLabel.Text = sprintf('Error loading audio: %s', ME.message);
                app.GenderLabel.Text = 'Gender: —';
                app.AgeLabel.Text    = 'Age: —';
                app.audioData = [];
                app.sampleRate = [];
                app.PlayButton.Enable = 'off';
            end
        end
        
        function PlayButtonPushed(app, ~)
            if ~isempty(app.audioData) && ~isempty(app.sampleRate)
                try
                    app.StatusLabel.Text = 'Playing audio...';
                    clear sound;
                    sound(app.audioData, app.sampleRate);
                    pause(0.2);
                    if ~isempty(app.audioFilename)
                        app.StatusLabel.Text = sprintf('Playing: %s', app.audioFilename);
                    else
                        app.StatusLabel.Text = 'Playing: microphone recording';
                    end
                catch ME
                    app.StatusLabel.Text = sprintf('Error playing audio: %s', ME.message);
                end
            else
                app.StatusLabel.Text = 'No audio loaded. Upload or record first.';
            end
        end
        
        % ===== Mic recording flow =====
        function RecordButtonPushed(app, ~)
            try
                if app.isRecording
                    app.StatusLabel.Text = 'Already recording...';
                    return;
                end
                fs = 16000;  % standardized
                app.recorder = audiorecorder(fs,16,1);
                record(app.recorder);
                app.isRecording = true;
                app.StatusLabel.Text = 'Recording... (press Stop when done)';
                app.RecordButton.Enable = 'off';
                app.StopButton.Enable   = 'on';
                app.UseRecordingButton.Enable = 'off';
            catch ME
                app.StatusLabel.Text = sprintf('Could not start recording: %s', ME.message);
            end
        end
        
        function StopButtonPushed(app, ~)
            try
                if ~app.isRecording || isempty(app.recorder)
                    app.StatusLabel.Text = 'Not recording right now.';
                    return;
                end
                stop(app.recorder);
                y = getaudiodata(app.recorder);
                if isempty(y)
                    app.StatusLabel.Text = 'No audio captured.';
                    app.isRecording = false;
                    app.RecordButton.Enable = 'on';
                    app.StopButton.Enable   = 'off';
                    return;
                end
                % Normalize and save
                y = y / max(1e-9, max(abs(y)));
                fs = app.recorder.SampleRate;
                
                outDir = 'D:\Jaffarproject\project\AudioPredictionProject\recordings';
                if ~exist(outDir,'dir'), mkdir(outDir); end
                fname = ['mic_' datestr(now,'yyyymmdd_HHMMSS') '.wav'];
                outPath = fullfile(outDir, fname);
                audiowrite(outPath, y, fs);
                app.lastRecordingPath = outPath;
                
                % Load into app state and predict immediately
                app.audioData   = y;
                app.sampleRate  = fs;
                app.audioFilename = '';
                app.PlayButton.Enable = 'on';
                
                app.StatusLabel.Text = sprintf('Recorded %0.1f s @ %d Hz. Predicting...', numel(y)/fs, fs);
                app.processAudio();
                
                % Restore buttons
                app.isRecording = false;
                app.RecordButton.Enable = 'on';
                app.StopButton.Enable   = 'off';
                app.UseRecordingButton.Enable = 'on';
                
            catch ME
                app.isRecording = false;
                app.RecordButton.Enable = 'on';
                app.StopButton.Enable   = 'off';
                app.UseRecordingButton.Enable = 'on';
                app.StatusLabel.Text = sprintf('Error stopping recording: %s', ME.message);
            end
        end
        
        function UseRecordingButtonPushed(app, ~)
            try
                if isempty(app.lastRecordingPath) || ~exist(app.lastRecordingPath,'file')
                    app.StatusLabel.Text = 'No saved recording found. Record first.';
                    return;
                end
                [y,fs] = audioread(app.lastRecordingPath);
                if size(y,2)>1, y = mean(y,2); end
                y = y / max(1e-9, max(abs(y)));
                app.audioData  = y;
                app.sampleRate = fs;
                app.audioFilename = '';
                app.PlayButton.Enable = 'on';
                
                app.StatusLabel.Text = 'Using last mic recording... Predicting...';
                app.processAudio();
            catch ME
                app.StatusLabel.Text = sprintf('Error loading last recording: %s', ME.message);
            end
        end
        
        % ===== Prediction pipeline (Gender + Age only) =====
        function processAudio(app)
            if isempty(app.audioData), return; end
            app.StatusLabel.Text = 'Processing audio...';
            drawnow;
            try
                [genderPred, agePred] = app.predictFromAudio(app.audioData, app.sampleRate);
                app.GenderLabel.Text = sprintf('Gender: %s', genderPred);
                app.AgeLabel.Text    = sprintf('Age: %s', agePred);
                app.StatusLabel.Text = 'Analysis complete!';
            catch ME
                app.StatusLabel.Text = sprintf('Processing error: %s', ME.message);
            end
        end
        
        function [gender, age] = predictFromAudio(app, audio, fs)
            gender = app.predictGender(audio, fs);
            age    = app.predictAge(audio, fs);
        end
        
        function gender = predictGender(~, audio, fs)
            try
                p = pitch(audio, fs);
                p = p(p>0);
                if isempty(p), gender = 'Unknown'; return; end
                if mean(p) > 180
                    gender = 'Female';
                else
                    gender = 'Male';
                end
            catch
                gender = 'Unknown';
            end
        end
        
        function age = predictAge(~, audio, fs)
            try
                p = pitch(audio, fs); p = p(p>0);
                if isempty(p), age = 'Unknown'; return; end
                avgPitch = mean(p);
                pitchVar = var(p);
                % Optional: centroid can be used later if you want
                try, sc = mean(spectralCentroid(audio, fs)); %#ok<NASGU>
                catch, end
                if     avgPitch > 220 && pitchVar > 800, age = '15-25 years';
                elseif avgPitch > 180 && pitchVar > 600, age = '20-30 years';
                elseif avgPitch > 160 && pitchVar > 400, age = '25-35 years';
                elseif avgPitch > 140 && pitchVar > 300, age = '30-45 years';
                elseif avgPitch > 120,                   age = '40-55 years';
                else                                      age = '50+ years';
                end
            catch
                age = 'Unknown';
            end
        end
        
        % ===== Responsive tweaks =====
        function onResize(app, ~)
            % Simple responsive behavior: increase title size on wide windows,
            % shrink on narrow; stretch button row.
            w = app.UIFigure.Position(3);
            if w >= 900
                app.TitleLabel.FontSize = 20;
                app.ButtonGrid.ColumnWidth = {'1x','1x','1x','1x','1x'};
            elseif w >= 650
                app.TitleLabel.FontSize = 18;
                app.ButtonGrid.ColumnWidth = {'1x','1x','1x','1x','1x'};
            else
                app.TitleLabel.FontSize = 16;
                % Slightly bias first two columns on narrow screens
                app.ButtonGrid.ColumnWidth = {'1x','1x','0.9x','0.9x','0.9x'};
            end
        end
    end
    
    methods (Access = private)
        function createComponents(app)
            % Figure
            app.UIFigure = uifigure('Visible','off');
            app.UIFigure.Position = [100 100 760 520];
            app.UIFigure.Name = 'Audio Age & Gender Prediction';
            app.UIFigure.Resize = 'on';
            app.UIFigure.SizeChangedFcn = @(~,~) app.onResize();
            
            % Root grid (top, middle, bottom)
            app.RootGrid = uigridlayout(app.UIFigure, [4,1]);
            app.RootGrid.RowHeight   = {60, 70, '1x', 34};
            app.RootGrid.ColumnWidth = {'1x'};
            app.RootGrid.Padding = [16 12 16 12];
            app.RootGrid.RowSpacing = 10;
            
            % Title
            app.TitleLabel = uilabel(app.RootGrid);
            app.TitleLabel.Layout.Row = 1;
            app.TitleLabel.Text = 'Audio Age & Gender Prediction';
            app.TitleLabel.HorizontalAlignment = 'center';
            app.TitleLabel.FontSize = 18;
            app.TitleLabel.FontWeight = 'bold';
            
            % Buttons line
            app.ButtonGrid = uigridlayout(app.RootGrid, [1,5]);
            app.ButtonGrid.Layout.Row = 2;
            app.ButtonGrid.ColumnWidth = {'1x','1x','1x','1x','1x'};
            app.ButtonGrid.ColumnSpacing = 8;
            app.ButtonGrid.RowSpacing = 0;
            app.ButtonGrid.Padding = [0 0 0 0];
            
            app.UploadButton = uibutton(app.ButtonGrid,'push','Text','Upload');
            app.UploadButton.Layout.Column = 1;
            app.UploadButton.FontSize = 14;
            app.UploadButton.ButtonPushedFcn = @(~,~) app.UploadButtonPushed();
            
            app.PlayButton = uibutton(app.ButtonGrid,'push','Text','Play');
            app.PlayButton.Layout.Column = 2;
            app.PlayButton.FontSize = 14;
            app.PlayButton.Enable = 'off';
            app.PlayButton.ButtonPushedFcn = @(~,~) app.PlayButtonPushed();
            
            app.RecordButton = uibutton(app.ButtonGrid,'push','Text','Record');
            app.RecordButton.Layout.Column = 3;
            app.RecordButton.FontSize = 14;
            app.RecordButton.ButtonPushedFcn = @(~,~) app.RecordButtonPushed();
            
            app.StopButton = uibutton(app.ButtonGrid,'push','Text','Stop');
            app.StopButton.Layout.Column = 4;
            app.StopButton.FontSize = 14;
            app.StopButton.Enable = 'off';
            app.StopButton.ButtonPushedFcn = @(~,~) app.StopButtonPushed();
            
            app.UseRecordingButton = uibutton(app.ButtonGrid,'push','Text','Use Last');
            app.UseRecordingButton.Layout.Column = 5;
            app.UseRecordingButton.FontSize = 14;
            app.UseRecordingButton.Enable = 'off';
            app.UseRecordingButton.ButtonPushedFcn = @(~,~) app.UseRecordingButtonPushed();
            
            % Results card
            app.ResultsPanel = uipanel(app.RootGrid);
            app.ResultsPanel.Layout.Row = 3;
            app.ResultsPanel.Title = 'Results';
            app.ResultsPanel.FontWeight = 'bold';
            app.ResultsPanel.BackgroundColor = [0.98 0.98 0.98];
            app.ResultsPanel.BorderType = 'line';
            
            app.ResultsGrid = uigridlayout(app.ResultsPanel,[2,1]);
            app.ResultsGrid.RowHeight   = {40, 40};
            app.ResultsGrid.ColumnWidth = {'1x'};
            app.ResultsGrid.Padding = [16 12 16 12];
            app.ResultsGrid.RowSpacing = 10;
            
            app.GenderLabel = uilabel(app.ResultsGrid);
            app.GenderLabel.Layout.Row = 1;
            app.GenderLabel.Text = 'Gender: Not analyzed';
            app.GenderLabel.FontSize = 16;
            app.GenderLabel.HorizontalAlignment = 'center';
            
            app.AgeLabel = uilabel(app.ResultsGrid);
            app.AgeLabel.Layout.Row = 2;
            app.AgeLabel.Text = 'Age: Not analyzed';
            app.AgeLabel.FontSize = 16;
            app.AgeLabel.HorizontalAlignment = 'center';
            
            % Status bar
            app.StatusLabel = uilabel(app.RootGrid);
            app.StatusLabel.Layout.Row = 4;
            app.StatusLabel.Text = 'Loading...';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.HorizontalAlignment = 'center';
            app.StatusLabel.FontColor = [0.35 0.35 0.35];
            
            app.UIFigure.Visible = 'on';
            
            % Initial responsive pass:
            app.onResize();
        end
    end
    
    % Public wrappers (so your earlier start/stop/predict commands still work)
    methods (Access = public)
        function app = AudioPredictionApp
            createComponents(app);
            registerApp(app, app.UIFigure);
            runStartupFcn(app,@startupFcn);
            if nargout == 0, clear app; end
        end
        
        function StartMicRecording(app)
            app.RecordButtonPushed();
        end
        function StopMicRecording(app)
            app.StopButtonPushed();
        end
        function UseLastRecordingAndPredict(app)
            app.UseRecordingButtonPushed();
        end
        
        function delete(app)
            try
                if app.isRecording && ~isempty(app.recorder)
                    stop(app.recorder);
                end
            catch
            end
            delete(app.UIFigure);
        end
    end
end
