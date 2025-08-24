function testModel()
    fprintf('=== Testing Audio Prediction Model ===\n');
    
    % Load model
    modelPath = 'D:\Jaffarproject\project\AudioPredictionProject\models\accentModel.mat';
    if ~exist(modelPath, 'file')
        fprintf('Error: Accent model not found at: %s\n', modelPath);
        fprintf('Run trainAccentModel() first.\n');
        return;
    end
    
    loaded = load(modelPath);
    model = loaded.accentModel;
    dialects = loaded.dialectLabels;
    
    fprintf('Model loaded successfully.\n');
    fprintf('Model type: %s\n', class(model));
    fprintf('Dialect regions:\n');
    for i = 1:length(dialects)
        fprintf('  DR%d: %s\n', i, dialects{i});
    end
    
    % Test with sample from dataset
    testPath = 'D:\Jaffarproject\project\AudioPredictionProject\datasets\timit\test\DR1';
    if ~exist(testPath, 'dir')
        fprintf('Test dataset not found at: %s\n', testPath);
        return;
    end
    
    fprintf('\nTesting with files from: %s\n', testPath);
    
    speakers = dir(testPath);
    speakers = speakers([speakers.isdir] & ~startsWith({speakers.name}, '.'));
    
    if isempty(speakers)
        fprintf('No speakers found in test directory.\n');
        return;
    end
    
    % Test first speaker
    speakerName = speakers(1).name;
    speakerPath = fullfile(testPath, speakerName);
    fprintf('Testing speaker: %s\n', speakerName);
    
    % Find WAV files
    wavFiles = dir(fullfile(speakerPath, '*.WAV'));
    if isempty(wavFiles)
        wavFiles = dir(fullfile(speakerPath, '*.wav'));
    end
    
    if isempty(wavFiles)
        fprintf('No WAV files found for speaker %s\n', speakerName);
        return;
    end
    
    % Test first file
    testFile = fullfile(speakerPath, wavFiles(1).name);
    fprintf('\nTesting with: %s\n', testFile);
    
    try
        % Load audio
        [audio, fs] = audioread(testFile);
        fprintf('Audio loaded: %.2f seconds, %d Hz\n', length(audio)/fs, fs);
        
        % Convert to mono if stereo
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        
        % Normalize
        audio = audio / max(abs(audio) + eps);
        
        % Test gender prediction
        pitchValues = pitch(audio, fs);
        pitchValues = pitchValues(pitchValues > 0 & pitchValues < 500); % Filter valid pitch
        
        if ~isempty(pitchValues)
            avgPitch = mean(pitchValues);
            pitchVar = var(pitchValues);
            
            % Updated age thresholds based on pitch
            if avgPitch > 220
                ageRange = '15-25 years';
            elseif avgPitch > 200
                ageRange = '20-30 years';
            elseif avgPitch > 180
                ageRange = '25-35 years';
            elseif avgPitch > 160
                ageRange = '30-45 years';
            elseif avgPitch > 140
                ageRange = '40-55 years';
            else
                ageRange = '50+ years';
            end
            
            if avgPitch > 180
                gender = 'Female';
            else
                gender = 'Male';
            end
            
            fprintf('Predicted Gender: %s (Avg Pitch: %.1f Hz, Variance: %.1f)\n', gender, avgPitch, pitchVar);
            fprintf('Predicted Age: %s\n', ageRange);
        else
            fprintf('Could not extract pitch information.\n');
        end
        
        % Test accent prediction using SAME extraction as training
        mfccFeats = extractMFCCFeatures(audio, fs);
        
        if isempty(mfccFeats)
            fprintf('Could not extract MFCC features for accent prediction.\n');
            return;
        end
        
        fprintf('MFCC feature dimensions: %d\n', length(mfccFeats));
        
        % Predict accent
        if isa(model, 'ClassificationECOC') || isa(model, 'ClassificationSVM')
            [prediction, scores] = predict(model, mfccFeats);
            
            fprintf('\n=== ACCENT PREDICTION RESULTS ===\n');
            fprintf('Predicted Accent: %s (DR%d) - Confidence: %.1f%%\n', ...
                dialects{prediction}, prediction, max(scores)*100);
            fprintf('Expected: New England (DR1)\n');
            
            fprintf('\nAll prediction scores:\n');
            for i = 1:length(dialects)
                if i <= length(scores)
                    fprintf('  DR%d %s: %.3f (%.1f%%)\n', i, dialects{i}, scores(i), scores(i)*100);
                end
            end
            
        elseif isa(model, 'ClassificationKNN')
            prediction = predict(model, mfccFeats);
            fprintf('\n=== ACCENT PREDICTION RESULTS ===\n');
            fprintf('Predicted Accent: %s (DR%d) - k-NN Model\n', dialects{prediction}, prediction);
            fprintf('Expected: New England (DR1)\n');
        else
            fprintf('Unknown model type: %s\n', class(model));
        end
        
    catch ME
        fprintf('Error testing: %s\n', ME.message);
        fprintf('Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
end

% Same MFCC extraction function as training
function mfccFeats = extractMFCCFeatures(audio, fs)
    try
        % Ensure minimum audio length
        if length(audio) < 0.1 * fs % Less than 100ms
            mfccFeats = [];
            return;
        end
        
        % Resample if needed
        targetFs = 16000;
        if fs ~= targetFs
            audio = resample(audio, targetFs, fs);
            fs = targetFs;
        end
        
        % Create analysis window
        windowLength = round(0.025 * fs); % 25ms window
        overlapLength = round(0.01 * fs);  % 10ms overlap
        window = hamming(windowLength);
        
        % Extract MFCC with updated syntax
        [coeffs, ~] = mfcc(audio, fs, ...
            'NumCoeffs', 13, ...
            'Window', window, ...
            'OverlapLength', overlapLength, ...
            'LogEnergy', 'Replace');
        
        % Check if coeffs is valid
        if isempty(coeffs) || size(coeffs, 1) < 2
            mfccFeats = [];
            return;
        end
        
        % Remove any NaN or Inf values
        coeffs = coeffs(~any(isnan(coeffs) | isinf(coeffs), 2), :);
        
        if isempty(coeffs)
            mfccFeats = [];
            return;
        end
        
        % Statistical features (SAME AS TRAINING)
        mfccMean = mean(coeffs, 1);
        mfccStd = std(coeffs, 0, 1);
        
        % Additional features for better classification (SAME AS TRAINING)
        mfccMin = min(coeffs, [], 1);
        mfccMax = max(coeffs, [], 1);
        
        % Combine all features (52 total: 13*4)
        mfccFeats = [mfccMean, mfccStd, mfccMin, mfccMax];
        
        % Ensure no NaN or Inf in final features
        if any(isnan(mfccFeats) | isinf(mfccFeats))
            mfccFeats = [];
        end
        
    catch ME
        fprintf('MFCC extraction error: %s\n', ME.message);
        mfccFeats = [];
    end
end