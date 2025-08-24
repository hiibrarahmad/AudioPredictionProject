classdef AudioPredictionApp < matlab.apps.AppBase
    
    properties (Access = public)
        UIFigure          matlab.ui.Figure
        GridLayout        matlab.ui.container.GridLayout
        TitleLabel        matlab.ui.control.Label
        UploadButton      matlab.ui.control.Button
        PlayButton        matlab.ui.control.Button
        GenderLabel       matlab.ui.control.Label
        AgeLabel          matlab.ui.control.Label
        AccentLabel       matlab.ui.control.Label
        StatusLabel       matlab.ui.control.Label
        
        % Properties for audio processing
        audioData         double
        sampleRate        double
        audioFilename     char
        accentModel       
        dialectLabels     cell
        featureValid      logical  % Track which features are valid
        expectedFeatures  double   % Expected number of features
    end
    
    methods (Access = private)
        
        function startupFcn(app)
            % Load accent model
            app.loadAccentModel();
            app.StatusLabel.Text = 'Ready. Please upload an audio file.';
        end
        
        function loadAccentModel(app)
            try
                modelPath = 'D:\Jaffarproject\project\AudioPredictionProject\models\accentModel.mat';
                if exist(modelPath, 'file')
                    loaded = load(modelPath);
                    app.accentModel = loaded.accentModel;
                    app.dialectLabels = loaded.dialectLabels;
                    
                    % Load feature validation info
                    if isfield(loaded, 'featureValid')
                        app.featureValid = loaded.featureValid;
                        app.expectedFeatures = sum(loaded.featureValid);
                        fprintf('Model expects %d features (out of %d total)\n', ...
                            app.expectedFeatures, length(loaded.featureValid));
                    else
                        % Fallback for older models - try to detect expected features
                        try
                            % Get expected features from model structure
                            if isprop(app.accentModel, 'NumPredictors')
                                app.expectedFeatures = app.accentModel.NumPredictors;
                            elseif isfield(app.accentModel, 'X')
                                app.expectedFeatures = size(app.accentModel.X, 2);
                            else
                                app.expectedFeatures = [];
                            end
                            app.featureValid = [];
                            fprintf('Warning: Feature validation info not found. Expected features: %d\n', ...
                                app.expectedFeatures);
                        catch
                            app.expectedFeatures = [];
                            app.featureValid = [];
                        end
                    end
                    
                    app.StatusLabel.Text = 'Accent model loaded successfully.';
                else
                    app.StatusLabel.Text = 'Warning: Accent model not found. Train model first.';
                    app.accentModel = [];
                    app.dialectLabels = {
                        'New England', 'Northern', 'North Midland', 'South Midland',
                        'Southern', 'New York City', 'Western', 'Army Brat'
                    };
                end
            catch ME
                app.StatusLabel.Text = sprintf('Error loading accent model: %s', ME.message);
                app.accentModel = [];
            end
        end
        
        function UploadButtonPushed(app, event)
            % File dialog
            [file, path] = uigetfile({'*.wav;*.WAV;*.mp3', 'Audio Files (*.wav, *.mp3)'}, 'Select Audio File');
            
            if isequal(file, 0) || isequal(path, 0)
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
                % Read audio file
                [audioTemp, fsTemp] = audioread(app.audioFilename);
                
                % Validate audio data
                if isempty(audioTemp)
                    error('Audio file is empty or corrupted');
                end
                
                % Convert to mono if stereo
                if size(audioTemp, 2) > 1
                    audioTemp = mean(audioTemp, 2);
                    fprintf('Converted stereo to mono\n');
                end
                
                % Normalize audio
                maxVal = max(abs(audioTemp));
                if maxVal > 0
                    audioTemp = audioTemp / maxVal;
                else
                    error('Audio contains only silence');
                end
                
                % Store the processed audio
                app.audioData = audioTemp;
                app.sampleRate = fsTemp;
                
                app.StatusLabel.Text = sprintf('Audio loaded: %s (%.1fs, %dHz)', file, length(audioTemp)/fsTemp, fsTemp);
                app.PlayButton.Enable = 'on';
                
                % Process audio for predictions
                app.processAudio();
                
            catch ME
                app.StatusLabel.Text = sprintf('Error loading audio: %s', ME.message);
                app.GenderLabel.Text = 'Gender: Error loading file';
                app.AgeLabel.Text = 'Age: Error loading file';
                app.AccentLabel.Text = 'Accent: Error loading file';
                app.audioData = [];
                app.sampleRate = [];
                app.PlayButton.Enable = 'off';
            end
        end
        
        function PlayButtonPushed(app, event)
            if ~isempty(app.audioData) && ~isempty(app.sampleRate)
                try
                    app.StatusLabel.Text = 'Playing audio...';
                    
                    % Clear any existing audio players
                    clear sound;
                    
                    % Play the loaded audio data
                    sound(app.audioData, app.sampleRate);
                    
                    % Update status after a short delay
                    pause(0.5);
                    app.StatusLabel.Text = sprintf('Playing: %s', app.audioFilename);
                    
                catch ME
                    app.StatusLabel.Text = sprintf('Error playing audio: %s', ME.message);
                end
            else
                app.StatusLabel.Text = 'No audio loaded. Please upload an audio file first.';
            end
        end
        
        function processAudio(app)
            if isempty(app.audioData)
                return;
            end
            
            app.StatusLabel.Text = 'Processing audio...';
            
            try
                % Extract features
                [genderPred, agePred, accentPred] = app.predictFromAudio(app.audioData, app.sampleRate);
                
                % Update labels
                app.GenderLabel.Text = sprintf('Gender: %s', genderPred);
                app.AgeLabel.Text = sprintf('Age: %s', agePred);
                app.AccentLabel.Text = sprintf('Accent: %s', accentPred);
                
                app.StatusLabel.Text = 'Analysis complete!';
                
            catch ME
                app.StatusLabel.Text = sprintf('Processing error: %s', ME.message);
            end
        end
        
        function [gender, age, accent] = predictFromAudio(app, audio, fs)
            % Gender prediction based on pitch
            gender = app.predictGender(audio, fs);
            
            % Age prediction based on pitch and spectral features
            age = app.predictAge(audio, fs);
            
            % Accent prediction using trained model
            accent = app.predictAccent(audio, fs);
        end
        
        function gender = predictGender(app, audio, fs)
            try
                % Extract pitch
                pitchValues = pitch(audio, fs);
                pitchValues = pitchValues(pitchValues > 0); % Remove unvoiced
                
                if isempty(pitchValues)
                    gender = 'Unknown';
                    return;
                end
                
                avgPitch = mean(pitchValues);
                
                % Gender classification thresholds
                if avgPitch > 180
                    gender = 'Female';
                else
                    gender = 'Male';
                end
                
            catch
                gender = 'Unknown';
            end
        end
        
        function age = predictAge(app, audio, fs)
            try
                % Extract pitch
                pitchValues = pitch(audio, fs);
                pitchValues = pitchValues(pitchValues > 0);
                
                if isempty(pitchValues)
                    age = 'Unknown';
                    return;
                end
                
                avgPitch = mean(pitchValues);
                pitchVar = var(pitchValues);
                
                % Calculate spectral centroid for additional age info
                try
                    spectralCentroid = mean(spectralCentroid(audio, fs));
                catch
                    spectralCentroid = 0;
                end
                
                % Improved age classification based on research
                % Younger people tend to have higher pitch variation and spectral energy
                if avgPitch > 220 && pitchVar > 800  % Very high pitch + high variation
                    age = '15-25 years';
                elseif avgPitch > 180 && pitchVar > 600  % High pitch + good variation
                    age = '20-30 years';
                elseif avgPitch > 160 && pitchVar > 400  % Medium-high pitch
                    age = '25-35 years';
                elseif avgPitch > 140 && pitchVar > 300  % Medium pitch
                    age = '30-45 years';
                elseif avgPitch > 120  % Lower pitch
                    age = '40-55 years';
                else
                    age = '50+ years';
                end
                
            catch
                age = 'Unknown';
            end
        end
        
        function accent = predictAccent(app, audio, fs)
            try
                if isempty(app.accentModel)
                    % Fallback: Basic accent detection based on audio characteristics
                    pitchValues = pitch(audio, fs);
                    pitchValues = pitchValues(pitchValues > 0);
                    
                    if isempty(pitchValues)
                        accent = 'Unknown - Model not available';
                        return;
                    end
                    
                    avgPitch = mean(pitchValues);
                    pitchStd = std(pitchValues);
                    
                    % Simple heuristic for non-US accents
                    if pitchStd > 50 && avgPitch > 180
                        accent = 'Likely Non-US (High variation)';
                    elseif pitchStd < 30
                        accent = 'Likely General American';
                    else
                        accent = 'Unknown Region';
                    end
                    return;
                end
                
                % Extract comprehensive features (MATCHING TRAINING EXACTLY)
                features = app.extractComprehensiveFeatures(audio, fs);
                
                if isempty(features)
                    accent = 'Feature extraction failed';
                    return;
                end
                
                fprintf('Raw features extracted: %d dimensions\n', length(features));
                
                % Apply feature selection if available
                if ~isempty(app.featureValid)
                    if length(features) ~= length(app.featureValid)
                        fprintf('Warning: Feature length mismatch. Expected %d, got %d\n', ...
                            length(app.featureValid), length(features));
                        % Try to match as best as possible
                        minLen = min(length(features), length(app.featureValid));
                        features = features(1:minLen);
                        featureValid = app.featureValid(1:minLen);
                        features = features(featureValid);
                    else
                        features = features(app.featureValid);
                    end
                    fprintf('Features after selection: %d dimensions\n', length(features));
                end
                
                % Final feature validation
                if ~isempty(app.expectedFeatures) && length(features) ~= app.expectedFeatures
                    fprintf('Error: Final feature dimension mismatch. Expected %d, got %d\n', ...
                        app.expectedFeatures, length(features));
                    accent = sprintf('Feature mismatch (got %d, need %d)', length(features), app.expectedFeatures);
                    return;
                end
                
                % Predict using trained model
                try
                    [prediction, scores] = predict(app.accentModel, features);
                    
                    % Check prediction validity
                    if prediction >= 1 && prediction <= length(app.dialectLabels)
                        % Get confidence (highest score)
                        maxScore = max(scores);
                        confidence = maxScore * 100;
                        
                        if confidence < 40 % Low confidence
                            accent = sprintf('%s (Low confidence: %.1f%%)', app.dialectLabels{prediction}, confidence);
                        else
                            accent = sprintf('%s (Confidence: %.1f%%)', app.dialectLabels{prediction}, confidence);
                        end
                    else
                        accent = 'Unknown region';
                    end
                    
                catch predError
                    fprintf('Prediction error: %s\n', predError.message);
                    accent = sprintf('Prediction failed: %s', predError.message);
                end
                
            catch ME
                fprintf('Accent prediction error: %s\n', ME.message);
                accent = 'Error in prediction';
            end
        end
        
        % COMPREHENSIVE FEATURE EXTRACTION (MATCHING TRAINING EXACTLY)
        function features = extractComprehensiveFeatures(app, audio, fs)
            try
                features = [];
                
                % Ensure minimum length
                if length(audio) < 0.2 * fs
                    return;
                end
                
                % Resample to standard rate
                targetFs = 16000;
                if fs ~= targetFs
                    audio = resample(audio, targetFs, fs);
                    fs = targetFs;
                end
                
                % For single audio file, simulate the training approach
                % Training uses multiple files per speaker, we'll segment the audio
                audioLength = length(audio);
                segmentLength = round(2 * fs); % 2-second segments
                overlap = round(0.5 * fs);     % 0.5-second overlap
                
                allSegmentFeatures = [];
                
                % Extract features from multiple segments (like training does with multiple files)
                startPos = 1;
                segmentCount = 0;
                
                while startPos + segmentLength <= audioLength && segmentCount < 6
                    endPos = startPos + segmentLength - 1;
                    segment = audio(startPos:endPos);
                    
                    % Extract base features for this segment
                    baseFeatures = app.extractBaseFeatures(segment, fs);
                    
                    if ~isempty(baseFeatures) && ~any(isnan(baseFeatures) | isinf(baseFeatures))
                        allSegmentFeatures = [allSegmentFeatures; baseFeatures];
                        segmentCount = segmentCount + 1;
                    end
                    
                    startPos = startPos + segmentLength - overlap;
                end
                
                % If we don't have enough segments, use the whole audio
                if size(allSegmentFeatures, 1) < 2
                    baseFeatures = app.extractBaseFeatures(audio, fs);
                    if ~isempty(baseFeatures)
                        % Duplicate to simulate multiple files
                        allSegmentFeatures = [baseFeatures; baseFeatures];
                    else
                        return;
                    end
                end
                
                % Now do the same aggregation as training:
                % [speakerMeanFeats, speakerStdFeats, speakerMinFeats, speakerMaxFeats]
                speakerMeanFeats = mean(allSegmentFeatures, 1);
                speakerStdFeats = std(allSegmentFeatures, 0, 1);
                speakerMinFeats = min(allSegmentFeatures, [], 1);
                speakerMaxFeats = max(allSegmentFeatures, [], 1);
                
                % Combine exactly like training
                features = [speakerMeanFeats, speakerStdFeats, speakerMinFeats, speakerMaxFeats];
                
                fprintf('Segments used: %d, Base features: %d, Final features: %d\n', ...
                    size(allSegmentFeatures, 1), length(speakerMeanFeats), length(features));
                
                % Final validation
                if any(isnan(features) | isinf(features))
                    features = [];
                end
                
            catch ME
                fprintf('Comprehensive feature extraction error: %s\n', ME.message);
                features = [];
            end
        end
        
        % Extract base features (single file features like training does)
        function features = extractBaseFeatures(app, audio, fs)
            try
                % 1. ENHANCED MFCC FEATURES
                mfccFeatures = app.extractEnhancedMFCC(audio, fs);
                if isempty(mfccFeatures)
                    features = [];
                    return;
                end
                
                % 2. PROSODIC FEATURES (crucial for accents)
                prosodicFeatures = app.extractProsodicFeatures(audio, fs);
                
                % 3. SPECTRAL FEATURES
                spectralFeatures = app.extractSpectralFeatures(audio, fs);
                
                % 4. FORMANT-RELATED FEATURES (vowel characteristics)
                formantFeatures = app.extractFormantFeatures(audio, fs);
                
                % 5. RHYTHM AND TIMING FEATURES
                rhythmFeatures = app.extractRhythmFeatures(audio, fs);
                
                % Combine all base features (137 dimensions)
                features = [mfccFeatures, prosodicFeatures, spectralFeatures, ...
                           formantFeatures, rhythmFeatures];
                
                % Final validation
                if any(isnan(features) | isinf(features))
                    features = [];
                end
                
            catch ME
                fprintf('Base feature extraction error: %s\n', ME.message);
                features = [];
            end
        end
        
        % Enhanced MFCC with delta and delta-delta (EXACT COPY FROM TRAINING)
        function mfccFeats = extractEnhancedMFCC(app, audio, fs)
            try
                windowLength = round(0.025 * fs);
                overlapLength = round(0.01 * fs);
                window = hamming(windowLength);
                
                % Extract MFCC coefficients
                [coeffs, ~] = mfcc(audio, fs, ...
                    'NumCoeffs', 13, ...
                    'Window', window, ...
                    'OverlapLength', overlapLength, ...
                    'LogEnergy', 'Replace');
                
                if size(coeffs, 1) < 3
                    mfccFeats = [];
                    return;
                end
                
                % Calculate delta (velocity) and delta-delta (acceleration)
                delta = [zeros(1, size(coeffs, 2)); diff(coeffs)];
                deltaDelta = [zeros(1, size(coeffs, 2)); diff(delta)];
                
                % Remove outliers
                coeffs = coeffs(~any(isnan(coeffs) | isinf(coeffs), 2), :);
                delta = delta(~any(isnan(delta) | isinf(delta), 2), :);
                deltaDelta = deltaDelta(~any(isnan(deltaDelta) | isinf(deltaDelta), 2), :);
                
                if isempty(coeffs)
                    mfccFeats = [];
                    return;
                end
                
                % Statistical measures for each feature type
                mfccMean = mean(coeffs, 1);
                mfccStd = std(coeffs, 0, 1);
                deltaMean = mean(delta, 1);
                deltaStd = std(delta, 0, 1);
                deltaDeltaMean = mean(deltaDelta, 1);
                deltaDeltaStd = std(deltaDelta, 0, 1);
                
                % Additional statistics
                mfccSkew = skewness(coeffs, 1);
                mfccKurt = kurtosis(coeffs, 1);
                
                mfccFeats = [mfccMean, mfccStd, deltaMean, deltaStd, ...
                            deltaDeltaMean, deltaDeltaStd, mfccSkew, mfccKurt];
                
            catch
                mfccFeats = [];
            end
        end
        
        % Prosodic features (EXACT COPY FROM TRAINING)
        function prosodicFeats = extractProsodicFeatures(app, audio, fs)
            try
                % Extract pitch contour
                pitchValues = pitch(audio, fs);
                pitchValues = pitchValues(pitchValues > 0 & pitchValues < 500);
                
                if length(pitchValues) < 10
                    prosodicFeats = zeros(1, 12);
                    return;
                end
                
                % Pitch statistics
                pitchMean = mean(pitchValues);
                pitchStd = std(pitchValues);
                pitchRange = max(pitchValues) - min(pitchValues);
                pitchMedian = median(pitchValues);
                
                % Pitch dynamics (important for accent detection)
                pitchSlope = polyfit(1:length(pitchValues), pitchValues', 1);
                pitchSlope = pitchSlope(1);
                
                % Jitter (pitch perturbation)
                if length(pitchValues) > 1
                    pitchJitter = mean(abs(diff(pitchValues))) / pitchMean;
                else
                    pitchJitter = 0;
                end
                
                % Contour shape features
                pitchRising = sum(diff(pitchValues) > 0) / length(pitchValues);
                pitchFalling = sum(diff(pitchValues) < 0) / length(pitchValues);
                
                % Energy-based features
                energy = audio.^2;
                energyMean = mean(energy);
                energyStd = std(energy);
                
                % Voice activity detection
                voicedRatio = length(pitchValues) / length(audio) * fs * 0.01; % Approximate
                
                % Rhythm indicators
                energyPeaks = findpeaks(energy);
                if ~isempty(energyPeaks)
                    rhythmRegularity = std(diff(energyPeaks)) / mean(diff(energyPeaks));
                else
                    rhythmRegularity = 1;
                end
                
                prosodicFeats = [pitchMean, pitchStd, pitchRange, pitchMedian, ...
                                pitchSlope, pitchJitter, pitchRising, pitchFalling, ...
                                energyMean, energyStd, voicedRatio, rhythmRegularity];
                
            catch
                prosodicFeats = zeros(1, 12);
            end
        end
        
        % Spectral features (EXACT COPY FROM TRAINING)
        function spectralFeats = extractSpectralFeatures(app, audio, fs)
            try
                windowLength = round(0.025 * fs);
                overlapLength = round(windowLength * 0.5);
                
                [S, F, T] = stft(audio, fs, 'Window', hamming(windowLength), ...
                    'OverlapLength', overlapLength);
                
                magnitude = abs(S);
                
                % Spectral centroid (brightness)
                centroid = sum(F .* magnitude) ./ sum(magnitude);
                centroidMean = mean(centroid(~isnan(centroid)));
                centroidStd = std(centroid(~isnan(centroid)));
                
                % Spectral rolloff
                cumMagnitude = cumsum(magnitude, 1);
                totalMagnitude = sum(magnitude, 1);
                rolloff85 = zeros(1, size(magnitude, 2));
                for t = 1:size(magnitude, 2)
                    idx = find(cumMagnitude(:, t) >= 0.85 * totalMagnitude(t), 1);
                    if ~isempty(idx)
                        rolloff85(t) = F(idx);
                    end
                end
                rolloffMean = mean(rolloff85(rolloff85 > 0));
                rolloffStd = std(rolloff85(rolloff85 > 0));
                
                % Spectral flatness (tonality measure)
                flatness = geomean(magnitude, 1) ./ mean(magnitude, 1);
                flatnessMean = mean(flatness(~isnan(flatness) & flatness > 0));
                flatnessStd = std(flatness(~isnan(flatness) & flatness > 0));
                
                % Zero crossing rate
                zcr = sum(abs(diff(sign(audio)))) / (2 * length(audio));
                
                % High-frequency content (important for fricatives)
                highFreqIdx = F > 4000;
                highFreqEnergy = mean(sum(magnitude(highFreqIdx, :), 1)) / mean(sum(magnitude, 1));
                
                spectralFeats = [centroidMean, centroidStd, rolloffMean, rolloffStd, ...
                                flatnessMean, flatnessStd, zcr, highFreqEnergy];
                
                % Handle NaN values
                spectralFeats(isnan(spectralFeats)) = 0;
                
            catch
                spectralFeats = zeros(1, 8);
            end
        end
        
        % Formant-related features (EXACT COPY FROM TRAINING)
        function formantFeats = extractFormantFeatures(app, audio, fs)
            try
                % Focus on vowel formant regions
                windowLength = round(0.025 * fs);
                overlapLength = round(windowLength * 0.5);
                
                [S, F, T] = stft(audio, fs, 'Window', hamming(windowLength), ...
                    'OverlapLength', overlapLength);
                
                magnitude = abs(S);
                
                % Define formant regions
                f1_region = F >= 200 & F <= 1000;   % F1 region
                f2_region = F >= 800 & F <= 2500;   % F2 region
                f3_region = F >= 2000 & F <= 4000;  % F3 region
                
                % Energy in formant regions
                f1_energy = mean(sum(magnitude(f1_region, :), 1));
                f2_energy = mean(sum(magnitude(f2_region, :), 1));
                f3_energy = mean(sum(magnitude(f3_region, :), 1));
                
                totalEnergy = mean(sum(magnitude, 1));
                
                f1_ratio = f1_energy / (totalEnergy + eps);
                f2_ratio = f2_energy / (totalEnergy + eps);
                f3_ratio = f3_energy / (totalEnergy + eps);
                
                % Formant bandwidth estimates
                f1_bw = std(sum(magnitude(f1_region, :), 1));
                f2_bw = std(sum(magnitude(f2_region, :), 1));
                
                % Vowel space indicators
                vowelCompactness = f1_ratio / (f2_ratio + eps);
                vowelFrontness = f2_ratio / (f1_ratio + eps);
                
                formantFeats = [f1_ratio, f2_ratio, f3_ratio, f1_bw, f2_bw, ...
                               vowelCompactness, vowelFrontness];
                
            catch
                formantFeats = zeros(1, 7);
            end
        end
        
        % Rhythm and timing features (EXACT COPY FROM TRAINING)
        function rhythmFeats = extractRhythmFeatures(app, audio, fs)
            try
                % Envelope extraction for rhythm analysis
                envelope = abs(hilbert(audio));
                envelope = smooth(envelope, round(fs * 0.01)); % 10ms smoothing
                
                % Find syllable-like units
                threshold = 0.3 * max(envelope);
                aboveThreshold = envelope > threshold;
                
                % Find onset times
                onsets = find(diff([0; aboveThreshold]) == 1);
                
                if length(onsets) < 3
                    rhythmFeats = zeros(1, 6);
                    return;
                end
                
                % Inter-onset intervals
                ioi = diff(onsets) / fs; % Convert to seconds
                
                % Rhythm metrics
                ioiMean = mean(ioi);
                ioiStd = std(ioi);
                ioiCV = ioiStd / (ioiMean + eps); % Coefficient of variation
                
                % Speech rate estimate
                speechRate = length(onsets) / (length(audio) / fs);
                
                % Pause analysis
                silenceThreshold = 0.1 * max(envelope);
                silences = envelope < silenceThreshold;
                pauseRatio = sum(silences) / length(silences);
                
                % Rhythmic regularity
                if length(ioi) > 1
                    rhythmRegularity = 1 / (1 + ioiCV); % Inverse of variability
                else
                    rhythmRegularity = 0.5;
                end
                
                rhythmFeats = [ioiMean, ioiStd, ioiCV, speechRate, pauseRatio, rhythmRegularity];
                
            catch
                rhythmFeats = zeros(1, 6);
            end
        end
    end
    
    methods (Access = private)
        
        function createComponents(app)
            % Create UIFigure
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 600 400];
            app.UIFigure.Name = 'Audio Age, Gender & Accent Prediction';
            app.UIFigure.Resize = 'off';
            
            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {'1x'};
            app.GridLayout.RowHeight = {50, 60, 60, 60, 60, 60, '1x'};
            
            % Create title
            app.TitleLabel = uilabel(app.GridLayout);
            app.TitleLabel.Layout.Row = 1;
            app.TitleLabel.Layout.Column = 1;
            app.TitleLabel.Text = 'Audio Age, Gender & Accent Prediction';
            app.TitleLabel.HorizontalAlignment = 'center';
            app.TitleLabel.FontSize = 18;
            app.TitleLabel.FontWeight = 'bold';
            
            % Create upload button
            app.UploadButton = uibutton(app.GridLayout, 'push');
            app.UploadButton.Layout.Row = 2;
            app.UploadButton.Layout.Column = 1;
            app.UploadButton.Text = 'Upload Audio File';
            app.UploadButton.FontSize = 14;
            app.UploadButton.ButtonPushedFcn = createCallbackFcn(app, @UploadButtonPushed, true);
            
            % Create play button
            app.PlayButton = uibutton(app.GridLayout, 'push');
            app.PlayButton.Layout.Row = 3;
            app.PlayButton.Layout.Column = 1;
            app.PlayButton.Text = 'Play Audio';
            app.PlayButton.FontSize = 14;
            app.PlayButton.Enable = 'off';
            app.PlayButton.ButtonPushedFcn = createCallbackFcn(app, @PlayButtonPushed, true);
            
            % Create result labels
            app.GenderLabel = uilabel(app.GridLayout);
            app.GenderLabel.Layout.Row = 4;
            app.GenderLabel.Layout.Column = 1;
            app.GenderLabel.Text = 'Gender: Not analyzed';
            app.GenderLabel.FontSize = 14;
            app.GenderLabel.HorizontalAlignment = 'center';
            
            app.AgeLabel = uilabel(app.GridLayout);
            app.AgeLabel.Layout.Row = 5;
            app.AgeLabel.Layout.Column = 1;
            app.AgeLabel.Text = 'Age: Not analyzed';
            app.AgeLabel.FontSize = 14;
            app.AgeLabel.HorizontalAlignment = 'center';
            
            app.AccentLabel = uilabel(app.GridLayout);
            app.AccentLabel.Layout.Row = 6;
            app.AccentLabel.Layout.Column = 1;
            app.AccentLabel.Text = 'Accent: Not analyzed';
            app.AccentLabel.FontSize = 14;
            app.AccentLabel.HorizontalAlignment = 'center';
            
            % Create status label
            app.StatusLabel = uilabel(app.GridLayout);
            app.StatusLabel.Layout.Row = 7;
            app.StatusLabel.Layout.Column = 1;
            app.StatusLabel.Text = 'Loading...';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.HorizontalAlignment = 'center';
            app.StatusLabel.FontColor = [0.5 0.5 0.5];
            
            % Show figure
            app.UIFigure.Visible = 'on';
        end
    end
    
    methods (Access = public)
        
        function app = AudioPredictionApp
            % Create components
            createComponents(app);
            
            % Register app with App Designer
            registerApp(app, app.UIFigure);
            
            % Execute startup function
            runStartupFcn(app, @startupFcn);
            
            if nargout == 0
                clear app;
            end
        end
        
        function delete(app)
            % Delete UIFigure when app is deleted
            delete(app.UIFigure);
        end
    end
end