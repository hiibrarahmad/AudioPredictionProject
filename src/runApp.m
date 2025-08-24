function runApp()
    fprintf('=== Audio Age, Gender & Accent Prediction ===\n');
    fprintf('Starting application...\n\n');
    
    % Check required toolboxes using ver instead of license
    requiredToolboxes = {'Audio Toolbox', 'Signal Processing Toolbox', 'Statistics and Machine Learning Toolbox'};
    installed = ver;
    
    fprintf('Checking required toolboxes...\n');
    for i = 1:length(requiredToolboxes)
        if any(strcmp({installed.Name}, requiredToolboxes{i}))
            fprintf('✓ %s\n', requiredToolboxes{i});
        else
            fprintf('✗ %s - NOT AVAILABLE\n', requiredToolboxes{i});
        end
    end
    fprintf('\n');
    
    % Create models directory
    modelsDir = 'D:\Jaffarproject\project\AudioPredictionProject\models';
    if ~exist(modelsDir, 'dir')
        mkdir(modelsDir);
        fprintf('Created models directory: %s\n', modelsDir);
    else
        fprintf('Models directory exists: %s\n', modelsDir);
    end
    
    % Check if accent model exists
    accentModelPath = fullfile(modelsDir, 'accentModel.mat');
    if ~exist(accentModelPath, 'file')
        fprintf('\n⚠️  Accent model not found at: %s\n', accentModelPath);
        fprintf('Run trainAccentModel() first to train the model.\n');
        fprintf('Or the app will use basic accent prediction.\n\n');
    else
        fprintf('✓ Accent model found at: %s\n\n', accentModelPath);
    end
    
    % Launch app
    fprintf('Launching GUI...\n');
    try
        app = AudioPredictionApp;
        fprintf('Application started successfully!\n');
        fprintf('\nInstructions:\n');
        fprintf('1. Click "Upload Audio File" to select a .wav or .mp3 file\n');
        fprintf('2. Click "Play Audio" to listen to the uploaded file\n');
        fprintf('3. View predictions for Gender, Age, and Accent\n');
    catch ME
        fprintf('Error starting application: %s\n', ME.message);
    end
end
