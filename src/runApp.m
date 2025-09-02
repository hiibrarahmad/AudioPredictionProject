function runApp()
    fprintf('=== Audio Age & Gender Prediction ===\n');
    fprintf('Starting application...\n\n');

    % Toolbox check
    requiredToolboxes = {'Audio Toolbox','Signal Processing Toolbox','Statistics and Machine Learning Toolbox'};
    installed = ver;
    fprintf('Checking required toolboxes...\n');
    for i = 1:numel(requiredToolboxes)
        if any(strcmp({installed.Name}, requiredToolboxes{i}))
            fprintf('✓ %s\n', requiredToolboxes{i});
        else
            fprintf('✗ %s - NOT AVAILABLE\n', requiredToolboxes{i});
        end
    end
    fprintf('\n');

    % Ensure recordings folder exists
    recDir = 'D:\Jaffarproject\project\AudioPredictionProject\recordings';
    if ~exist(recDir,'dir')
        mkdir(recDir);
        fprintf('Created recordings folder: %s\n', recDir);
    else
        fprintf('Recordings folder exists: %s\n', recDir);
    end

    % Launch GUI and expose base-workspace helpers (optional)
    fprintf('Launching GUI...\n');
    try
        app = AudioPredictionApp;
        fprintf('Application started. Use Upload/Record to analyze.\n');

        % Optional helpers to match your earlier workflow
        assignin('base','startRecording',@() app.StartMicRecording());
        assignin('base','stopRecording',@() app.StopMicRecording());
        assignin('base','predictRecording',@() app.UseLastRecordingAndPredict());

        fprintf('\nHelpers available:\n');
        fprintf('  startRecording();\n');
        fprintf('  stopRecording();\n');
        fprintf('  predictRecording(); %% uses last mic recording\n');

    catch ME
        fprintf('Error starting application: %s\n', ME.message);
    end
end
