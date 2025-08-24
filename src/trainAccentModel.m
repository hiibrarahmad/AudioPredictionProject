%% Enhanced TIMIT Accent Model Training Script
%% Advanced feature extraction and model training for better accent detection

fprintf('=== Enhanced TIMIT Accent Model Training ===\n');

% Define dataset paths
basePath = 'D:\Jaffarproject\project\AudioPredictionProject\datasets\timit';
trainPath = fullfile(basePath, 'train');
testPath = fullfile(basePath, 'test');

% Check which dataset exists
if exist(trainPath, 'dir')
    datasetPath = trainPath;
    fprintf('Using training dataset: %s\n', datasetPath);
elseif exist(testPath, 'dir')
    datasetPath = testPath;
    fprintf('Using test dataset: %s\n', datasetPath);
else
    error('TIMIT dataset not found. Checked:\n%s\n%s', trainPath, testPath);
end

% Enhanced dialect region mapping with characteristics
dialectLabels = {
    'New England',      % dr1 - Non-rhotic, broad 'a'
    'Northern',         % dr2 - Vowel shifting, nasal
    'North Midland',    % dr3 - General American base
    'South Midland',    % dr4 - Slight Southern influence
    'Southern',         % dr5 - Drawl, monophthongization
    'New York City',    % dr6 - Unique vowel system
    'Western',          % dr7 - Cot-caught merger
    'Army Brat'         % dr8 - Mixed/neutral
};

% Initialize storage for enhanced features
allFeatures = [];
allLabels = [];
allSpeakerIds = [];

fprintf('\n=== FEATURE EXTRACTION PHASE ===\n');

% Process each dialect region with enhanced sampling
for dr = 1:8
    drPath = fullfile(datasetPath, sprintf('DR%d', dr));
    if ~exist(drPath, 'dir')
        fprintf('Warning: DR%d not found, skipping...\n', dr);
        continue;
    end
    
    fprintf('Processing DR%d: %s\n', dr, dialectLabels{dr});
    
    % Get all speaker directories
    speakers = dir(drPath);
    speakers = speakers([speakers.isdir] & ~startsWith({speakers.name}, '.'));
    
    dialectFeatures = [];
    dialectLabelsTemp = [];
    speakerCount = 0;
    
    for s = 1:length(speakers)
        speakerPath = fullfile(drPath, speakers(s).name);
        speakerId = speakers(s).name;
        
        % Find all WAV files (use more files per speaker)
        wavFiles = dir(fullfile(speakerPath, '*.WAV'));
        if isempty(wavFiles)
            wavFiles = dir(fullfile(speakerPath, '*.wav'));
        end
        
        if isempty(wavFiles)
            continue;
        end
        
        speakerFeatures = [];
        validFiles = 0;
        
        % Process more files per speaker for better representation
        for w = 1:min(6, length(wavFiles)) % Increased from 3 to 6 files
            try
                wavFile = fullfile(speakerPath, wavFiles(w).name);
                
                % Read and preprocess audio
                [audio, fs] = audioread(wavFile);
                
                % Enhanced preprocessing
                audio = preprocessAudio(audio, fs);
                if isempty(audio)
                    continue;
                end
                
                % Extract comprehensive features
                features = extractComprehensiveFeatures(audio, fs);
                
                if ~isempty(features) && ~any(isnan(features) | isinf(features))
                    speakerFeatures = [speakerFeatures; features];
                    validFiles = validFiles + 1;
                end
                
            catch ME
                fprintf('  Warning: Failed %s - %s\n', wavFiles(w).name, ME.message);
            end
        end
        
        % Use speaker-level aggregation for more robust features
        if size(speakerFeatures, 1) >= 2
            % Aggregate features across files for this speaker
            speakerMeanFeats = mean(speakerFeatures, 1);
            speakerStdFeats = std(speakerFeatures, 0, 1);
            speakerMinFeats = min(speakerFeatures, [], 1);
            speakerMaxFeats = max(speakerFeatures, [], 1);
            
            % Combine speaker-level features
            finalFeatures = [speakerMeanFeats, speakerStdFeats, speakerMinFeats, speakerMaxFeats];
            
            dialectFeatures = [dialectFeatures; finalFeatures];
            dialectLabelsTemp = [dialectLabelsTemp; dr];
            speakerCount = speakerCount + 1;
            
            fprintf('  %s: %d files -> features extracted\n', speakerId, validFiles);
        end
        
        % Use more speakers per dialect for better coverage
        if speakerCount >= 30  % Increased from 20 to 30
            break;
        end
    end
    
    % Add to overall dataset
    allFeatures = [allFeatures; dialectFeatures];
    allLabels = [allLabels; dialectLabelsTemp];
    
    fprintf('  -> %d speakers, %d feature vectors\n', speakerCount, size(dialectFeatures, 1));
end

if isempty(allFeatures)
    error('No features extracted. Check dataset path and audio files.');
end

fprintf('\n=== DATASET STATISTICS ===\n');
fprintf('Total samples: %d\n', size(allFeatures, 1));
fprintf('Feature dimensions: %d\n', size(allFeatures, 2));

% Enhanced class distribution analysis
fprintf('Class distribution:\n');
classWeights = [];
for dr = 1:8
    count = sum(allLabels == dr);
    percentage = (count / length(allLabels)) * 100;
    fprintf('  DR%d (%s): %d samples (%.1f%%)\n', dr, dialectLabels{dr}, count, percentage);
    
    if count > 0
        classWeights(dr) = 1.0 / count;  % Inverse frequency weighting
    else
        classWeights(dr) = 0;
    end
end

% Normalize class weights
classWeights = classWeights / sum(classWeights) * length(unique(allLabels));

% Feature standardization and selection
fprintf('\n=== FEATURE PROCESSING ===\n');

% Remove constant or near-constant features
featureVar = var(allFeatures, 0, 1);
validFeatures = featureVar > 1e-6;
allFeatures = allFeatures(:, validFeatures);
fprintf('Removed %d constant features, %d remaining\n', ...
    sum(~validFeatures), sum(validFeatures));

% Check for class separability
uniqueLabels = unique(allLabels);
minSamples = min(arrayfun(@(x) sum(allLabels == x), uniqueLabels));

if minSamples < 5
    fprintf('Warning: Some classes have very few samples (min: %d)\n', minSamples);
end

% Train multiple models and select the best
fprintf('\n=== MODEL TRAINING ===\n');

bestModel = [];
bestAccuracy = 0;
bestModelType = '';

% Try different model configurations
models = {
    struct('name', 'SVM-RBF', 'kernel', 'rbf', 'scale', 'auto'), ...
    struct('name', 'SVM-Linear', 'kernel', 'linear', 'scale', 'auto'), ...
    struct('name', 'SVM-Polynomial', 'kernel', 'polynomial', 'scale', 'auto')
};

for m = 1:length(models)
    modelConfig = models{m};
    fprintf('Training %s classifier...\n', modelConfig.name);
    
    try
        % Create weighted SVM template
        template = templateSVM('Standardize', true, ...
            'KernelFunction', modelConfig.kernel, ...
            'KernelScale', modelConfig.scale, ...
            'BoxConstraint', 1.0);
        
        % Train with class weighting
        if length(unique(allLabels)) > 2
            accentModel = fitcecoc(allFeatures, allLabels, ...
                'Learners', template, ...
                'ClassNames', unique(allLabels), ...
                'Cost', createCostMatrix(allLabels, classWeights));
        else
            accentModel = fitcsvm(allFeatures, allLabels, template);
        end
        
        % Cross-validation for model evaluation
        cvModel = crossval(accentModel, 'KFold', 5);
        cvAccuracy = 1 - kfoldLoss(cvModel);
        
        fprintf('  Cross-validation accuracy: %.2f%%\n', cvAccuracy * 100);
        
        if cvAccuracy > bestAccuracy
            bestModel = accentModel;
            bestAccuracy = cvAccuracy;
            bestModelType = modelConfig.name;
        end
        
    catch ME
        fprintf('  Error with %s: %s\n', modelConfig.name, ME.message);
    end
end

% Fallback to k-NN if SVM fails
if isempty(bestModel)
    fprintf('Trying k-NN classifier as fallback...\n');
    try
        % Try different k values
        for k = [3, 5, 7, 9]
            knnModel = fitcknn(allFeatures, allLabels, ...
                'NumNeighbors', k, ...
                'Standardize', true, ...
                'Distance', 'euclidean');
            
            cvModel = crossval(knnModel, 'KFold', 5);
            cvAccuracy = 1 - kfoldLoss(cvModel);
            
            fprintf('  k-NN (k=%d) accuracy: %.2f%%\n', k, cvAccuracy * 100);
            
            if cvAccuracy > bestAccuracy
                bestModel = knnModel;
                bestAccuracy = cvAccuracy;
                bestModelType = sprintf('k-NN (k=%d)', k);
            end
        end
    catch ME
        fprintf('k-NN training failed: %s\n', ME.message);
        return;
    end
end

if isempty(bestModel)
    error('All model training attempts failed.');
end

fprintf('\nBest model: %s (CV accuracy: %.2f%%)\n', bestModelType, bestAccuracy * 100);

% Final model evaluation
fprintf('\n=== MODEL EVALUATION ===\n');
predictions = predict(bestModel, allFeatures);
trainingAccuracy = sum(predictions == allLabels) / length(allLabels);
fprintf('Training accuracy: %.2f%%\n', trainingAccuracy * 100);

% Per-class accuracy analysis
fprintf('Per-dialect accuracy:\n');
for dr = 1:8
    idx = allLabels == dr;
    if sum(idx) > 0
        drAccuracy = sum(predictions(idx) == dr) / sum(idx);
        fprintf('  DR%d (%s): %.1f%% (%d/%d)\n', dr, dialectLabels{dr}, ...
            drAccuracy*100, sum(predictions(idx) == dr), sum(idx));
    end
end

% Save enhanced model
fprintf('\n=== MODEL SAVING ===\n');
modelsPath = 'D:\Jaffarproject\project\AudioPredictionProject\models';
if ~exist(modelsPath, 'dir')
    mkdir(modelsPath);
    fprintf('Created models directory: %s\n', modelsPath);
end

% Save comprehensive model data
accentModel = bestModel;
modelType = bestModelType;
crossValAccuracy = bestAccuracy;
featureValid = validFeatures; % Save which features were used

modelFile = fullfile(modelsPath, 'accentModel.mat');
save(modelFile, 'accentModel', 'dialectLabels', 'modelType', ...
    'crossValAccuracy', 'featureValid');

fprintf('Enhanced accent model saved: %s\n', modelFile);
fprintf('Model type: %s\n', modelType);
fprintf('Cross-validation accuracy: %.2f%%\n', crossValAccuracy * 100);
fprintf('Feature dimensions: %d\n', sum(featureValid));

fprintf('\n=== TRAINING COMPLETE ===\n');

%% Enhanced audio preprocessing
function cleanAudio = preprocessAudio(audio, fs)
    try
        % Convert to mono
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        
        % Remove silence from beginning and end
        audioAbs = abs(audio);
        threshold = 0.01 * max(audioAbs);
        
        startIdx = find(audioAbs > threshold, 1, 'first');
        endIdx = find(audioAbs > threshold, 1, 'last');
        
        if isempty(startIdx) || isempty(endIdx) || (endIdx - startIdx) < 0.1 * fs
            cleanAudio = [];
            return;
        end
        
        audio = audio(startIdx:endIdx);
        
        % Normalize with soft limiting
        maxVal = max(abs(audio));
        if maxVal > 0
            audio = audio / (maxVal + eps);
            % Soft limiting to prevent clipping artifacts
            audio = tanh(audio * 0.9) / tanh(0.9);
        else
            cleanAudio = [];
            return;
        end
        
        cleanAudio = audio;
        
    catch
        cleanAudio = [];
    end
end

%% Comprehensive feature extraction for accent detection
function features = extractComprehensiveFeatures(audio, fs)
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
        
        % 1. ENHANCED MFCC FEATURES
        mfccFeatures = extractEnhancedMFCC(audio, fs);
        if isempty(mfccFeatures)
            return;
        end
        
        % 2. PROSODIC FEATURES (crucial for accents)
        prosodicFeatures = extractProsodicFeatures(audio, fs);
        
        % 3. SPECTRAL FEATURES
        spectralFeatures = extractSpectralFeatures(audio, fs);
        
        % 4. FORMANT-RELATED FEATURES (vowel characteristics)
        formantFeatures = extractFormantFeatures(audio, fs);
        
        % 5. RHYTHM AND TIMING FEATURES
        rhythmFeatures = extractRhythmFeatures(audio, fs);
        
        % Combine all features
        features = [mfccFeatures, prosodicFeatures, spectralFeatures, ...
                   formantFeatures, rhythmFeatures];
        
        % Final validation
        if any(isnan(features) | isinf(features))
            features = [];
        end
        
    catch
        features = [];
    end
end

%% Enhanced MFCC with delta and delta-delta
function mfccFeats = extractEnhancedMFCC(audio, fs)
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

%% Prosodic features (intonation, stress patterns)
function prosodicFeats = extractProsodicFeatures(audio, fs)
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

%% Spectral features
function spectralFeats = extractSpectralFeatures(audio, fs)
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

%% Formant-related features (vowel characteristics)
function formantFeats = extractFormantFeatures(audio, fs)
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

%% Rhythm and timing features
function rhythmFeats = extractRhythmFeatures(audio, fs)
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

%% Create cost matrix for imbalanced classes
function costMatrix = createCostMatrix(labels, classWeights)
    uniqueLabels = unique(labels);
    numClasses = length(uniqueLabels);
    costMatrix = ones(numClasses) - eye(numClasses);
    
    % Apply class weights to cost matrix
    for i = 1:numClasses
        for j = 1:numClasses
            if i ~= j
                costMatrix(i, j) = costMatrix(i, j) * classWeights(uniqueLabels(j));
            end
        end
    end
end