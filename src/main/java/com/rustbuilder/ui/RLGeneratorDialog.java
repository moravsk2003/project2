package com.rustbuilder.ui;

import com.rustbuilder.ai.rl.RLModelManager;
import com.rustbuilder.ai.rl.RLModelManager.RLModel;
import com.rustbuilder.ai.rl.RLTrainingService;
import com.rustbuilder.model.GridModel;

import java.util.List;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * Dialog for RL (Q-Learning) base generation.
 */
public class RLGeneratorDialog {

    private final GridModel gridModel;
    private final GameCanvas gameCanvas;
    private final Stage dialogStage;

    private RLTrainingService rlService;
    private String currentModelName;

    // UI Components
    private ComboBox<String> modelComboBox;
    private TextField newModelField;
    private Slider logisticsSlider;
    private Slider costSlider;
    private Slider raidSlider;
    private Label logLabel;
    private Label costLabel;
    private Label raidLabel;
    private Spinner<Integer> episodesSpinner;
    private Spinner<Integer> stepsSpinner;
    private Slider epochSlider;
    private Label epochLabel;
    private ProgressBar progressBar;
    private Label statusLabel;
    private Button trainButton;
    private Button generateButton;
    private Button deleteButton;
    private javafx.scene.control.CheckBox multiDiscreteCheck;

    // Stats
    private final Label rewardStatsLabel = new Label("🏆 Best Eval Score: —");
    private final Label avgEvalLabel = new Label("📈 Avg Eval Score (last 50): —");
    private final Label avgRewardLabel = new Label("📊 Avg Step Reward: —");
    private final Label invalidRateLabel = new Label("❌ Invalid Action Rate: —");
    private final Label memoryUsageLabel = new Label("💾 RAM Usage: —");
    private final Label bestBaseLabel = new Label("🧱 Best Base: —");
    private final Label qTableSizeLabel = new Label("🧠 Replay Memory Size: 0");
    private final Label episodeStatsLabel = new Label("🔁 Total Episodes: 0");
    private final Label epsilonStatsLabel = new Label("🎲 Epsilon (ε): —");
    private final Label lossStatsLabel = new Label("📉 Loss: —");
    private final Label timerLabel = new Label("⏱ Training time: —");
    private volatile boolean timerRunning = false;

    private static class BaseSnapshot {
        String name;
        GridModel model;
        BaseSnapshot(String n, GridModel m) { this.name = n; this.model = m; }
        @Override public String toString() { return name; }
    }
    private ComboBox<BaseSnapshot> baseHistoryCombo;
    
    private VBox mainContent;

    // shared styles
    private static final String CARD_STYLE =
        "-fx-background-color: #333; -fx-background-radius: 6; -fx-border-color: #4a4a4a; -fx-border-radius: 6; -fx-padding: 10 12 10 12;";
    private static final String SECTION_TITLE =
        "-fx-font-size: 13px; -fx-font-weight: bold; -fx-text-fill: #ddd;";
    private static final String BODY_LABEL =
        "-fx-text-fill: #bbb; -fx-font-size: 11px;";
    private static final String VALUE_LABEL =
        "-fx-text-fill: #e8e8e8; -fx-font-size: 11px; -fx-font-family: 'Consolas', monospace;";

    public RLGeneratorDialog(GridModel gridModel, GameCanvas gameCanvas, Stage owner) {
        this.gridModel = gridModel;
        this.gameCanvas = gameCanvas;

        dialogStage = new Stage();
        dialogStage.initModality(Modality.NONE);
        dialogStage.initOwner(owner);
        dialogStage.setTitle("🤖 Q-Learning AI");
        dialogStage.setResizable(true);

        mainContent = new VBox(10);
        mainContent.setPadding(new Insets(14));
        mainContent.setStyle("-fx-background-color: #2b2b2b;");

        mainContent.getChildren().addAll(
            createModelSection(),
            new Separator(),
            createPrioritySection(),
            new Separator(),
            createTrainingSection(),
            new Separator(),
            createExperimentalSection(),
            new Separator(),
            createStatsSection(),
            new Separator(),
            createActionSection()
        );

        ScrollPane scroll = new ScrollPane(mainContent);
        scroll.setFitToWidth(true);
        scroll.setStyle("-fx-background: #2b2b2b; -fx-background-color: #2b2b2b;");
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);

        Scene scene = new Scene(scroll, 460, 600);
        dialogStage.setScene(scene);
    }

    private VBox createModelSection() {
        VBox box = card();
        Label title = sectionTitle("📁  RL Model");

        HBox selectRow = new HBox(8);
        selectRow.setAlignment(Pos.CENTER_LEFT);
        modelComboBox = new ComboBox<>();
        modelComboBox.setPrefWidth(210);
        modelComboBox.setPromptText("Select model...");
        refreshModelList();

        Button loadBtn = styledBtn("Load", "#3498db");
        loadBtn.setOnAction(e -> loadSelectedModel());

        deleteButton = styledBtn("🗑", "#e74c3c");
        deleteButton.setOnAction(e -> deleteSelectedModel());

        selectRow.getChildren().addAll(modelComboBox, loadBtn, deleteButton);

        HBox newRow = new HBox(8);
        newRow.setAlignment(Pos.CENTER_LEFT);
        newModelField = new TextField();
        newModelField.setPromptText("New model name...");
        HBox.setHgrow(newModelField, Priority.ALWAYS);

        Button createBtn = styledBtn("+ Create", "#2ecc71");
        createBtn.setOnAction(e -> createNewModel());
        newRow.getChildren().addAll(newModelField, createBtn);

        box.getChildren().addAll(title, selectRow, newRow);
        return box;
    }

    private VBox createPrioritySection() {
        VBox box = card();
        Label title = sectionTitle("⚖  Priorities");

        logLabel = valueLabel("Logistics:        33%");
        logisticsSlider = buildSlider(0, 1, 0.33, 0.25);
        logisticsSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        costLabel = valueLabel("Resources:        33%");
        costSlider = buildSlider(0, 1, 0.33, 0.25);
        costSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        raidLabel = valueLabel("Raid Resistance:  34%");
        raidSlider = buildSlider(0, 1, 0.34, 0.25);
        raidSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        box.getChildren().addAll(title, logLabel, logisticsSlider, costLabel, costSlider, raidLabel, raidSlider);
        return box;
    }

    private void updateSliderLabels() {
        double sum = logisticsSlider.getValue() + costSlider.getValue() + raidSlider.getValue();
        if (sum < 0.001) sum = 1.0;
        logLabel.setText(String.format("Logistics:        %3.0f%%", logisticsSlider.getValue() / sum * 100));
        costLabel.setText(String.format("Resources:        %3.0f%%", costSlider.getValue() / sum * 100));
        raidLabel.setText(String.format("Raid Resistance:  %3.0f%%", raidSlider.getValue() / sum * 100));
    }

    private double[] getNormalizedWeights() {
        double sum = logisticsSlider.getValue() + costSlider.getValue() + raidSlider.getValue();
        if (sum < 0.001) return new double[]{0.33, 0.33, 0.34};
        return new double[]{ logisticsSlider.getValue() / sum, costSlider.getValue() / sum, raidSlider.getValue() / sum };
    }

    private VBox createTrainingSection() {
        VBox box = card();
        Label title = sectionTitle("🧠  Training");

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(6);

        Label epLabel = bodyLabel("Episodes per Epoch:");
        episodesSpinner = new Spinner<>(10, 10000, 500, 50);
        episodesSpinner.setPrefWidth(90);
        episodesSpinner.setEditable(true);
        Tooltip.install(episodesSpinner, new Tooltip("Number of episodes to train per epoch"));

        Label stepsLabel = bodyLabel("Max Steps/Episode:");
        stepsSpinner = new Spinner<>(10, 100, 40, 5);
        stepsSpinner.setPrefWidth(90);
        stepsSpinner.setEditable(true);
        Tooltip.install(stepsSpinner, new Tooltip("Maximum blocks the agent can place per episode"));

        grid.add(epLabel, 0, 0); grid.add(episodesSpinner, 1, 0);
        grid.add(stepsLabel, 0, 1); grid.add(stepsSpinner, 1, 1);

        epochLabel = valueLabel("Epochs: 1  (total: 500 episodes)");
        epochSlider = buildSlider(1, 20, 1, 5);
        epochSlider.setMinorTickCount(0);
        epochSlider.setSnapToTicks(true);
        epochSlider.setBlockIncrement(1);
        epochSlider.valueProperty().addListener((o, ov, nv) -> {
            int ep = (int) Math.round(nv.doubleValue());
            epochLabel.setText(String.format("Epochs: %d  (total: %d episodes)",
                ep, ep * episodesSpinner.getValue()));
        });

        trainButton = styledBtn("▶ Train", "#e67e22");
        trainButton.setStyle(trainButton.getStyle() + " -fx-font-weight: bold;");
        trainButton.setOnAction(e -> startTraining());

        progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(Double.MAX_VALUE);
        progressBar.setVisible(false);

        statusLabel = new Label("No model loaded. Create or load a model to start.");
        statusLabel.setStyle("-fx-text-fill: #888; -fx-font-size: 11px;");
        statusLabel.setWrapText(true);

        box.getChildren().addAll(title, grid, epochLabel, epochSlider, trainButton, progressBar, statusLabel);
        return box;
    }

    private VBox createExperimentalSection() {
        VBox box = card();
        Label title = sectionTitle("🧪  Experimental Flows");

        multiDiscreteCheck = new javafx.scene.control.CheckBox("Enable 5-Phase Multi-Discrete Flow");
        multiDiscreteCheck.setStyle("-fx-text-fill: #ecf0f1;");
        // Initialization: Use UI as truth if service not yet active
        multiDiscreteCheck.setSelected(true); 
        multiDiscreteCheck.setOnAction(e -> { 
            if (rlService != null) {
                rlService.setUseMultiDiscreteFlow(multiDiscreteCheck.isSelected());
            }
        });

        box.getChildren().addAll(title, multiDiscreteCheck);
        return box;
    }

    private VBox createStatsSection() {
        VBox box = card();
        box.setStyle(CARD_STYLE + " -fx-background-color: #2e2e2e;");
        Label title = sectionTitle("📊  Model Stats");
        rewardStatsLabel.setStyle(VALUE_LABEL);
        avgEvalLabel.setStyle(VALUE_LABEL);
        avgRewardLabel.setStyle(VALUE_LABEL);
        invalidRateLabel.setStyle(VALUE_LABEL);
        memoryUsageLabel.setStyle(VALUE_LABEL);
        bestBaseLabel.setStyle(VALUE_LABEL);
        qTableSizeLabel.setStyle(VALUE_LABEL);
        episodeStatsLabel.setStyle(VALUE_LABEL);
        epsilonStatsLabel.setStyle(VALUE_LABEL);
        timerLabel.setStyle(VALUE_LABEL);
        lossStatsLabel.setStyle(VALUE_LABEL);
        
        // Separator between key training metrics and model internals
        Separator statsSep = new Separator();
        statsSep.setStyle("-fx-padding: 4 0 4 0;");
        
        HBox baseRow = new HBox(10);
        baseRow.setAlignment(Pos.CENTER_LEFT);
        Label baseLabel = new Label("🏰 History:");
        baseLabel.setStyle(BODY_LABEL);
        baseHistoryCombo = new ComboBox<>();
        baseHistoryCombo.setPrefWidth(220);
        baseHistoryCombo.setPromptText("Generated bases...");
        baseHistoryCombo.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null && newVal.model != null) {
                gridModel.clear();
                for (com.rustbuilder.model.core.BuildingBlock b : newVal.model.getAllBlocks()) {
                    gridModel.addBlockSilent(b);
                }
                gridModel.finalizeLoad();
                gameCanvas.draw();
            }
        });
        baseRow.getChildren().addAll(baseLabel, baseHistoryCombo);

        box.getChildren().addAll(title,
            baseRow,
            rewardStatsLabel, avgEvalLabel, avgRewardLabel, invalidRateLabel,
            statsSep,
            bestBaseLabel, memoryUsageLabel,
            qTableSizeLabel, episodeStatsLabel, epsilonStatsLabel, lossStatsLabel, timerLabel);
        return box;
    }

    private HBox createActionSection() {
        HBox box = new HBox(10);
        box.setAlignment(Pos.CENTER);

        generateButton = styledBtn("🏠 Generate Best", "#9b59b6");
        generateButton.setStyle(generateButton.getStyle() + " -fx-font-weight: bold; -fx-font-size: 13px;");
        generateButton.setPrefWidth(140);
        generateButton.setDisable(true);
        generateButton.setOnAction(e -> generateBest());

        Button saveBtn = styledBtn("💾 Save", "#1abc9c");
        saveBtn.setOnAction(e -> saveCurrentModel());

        Button closeBtn = styledBtn("✕ Close", "#555");
        closeBtn.setOnAction(e -> dialogStage.close());

        box.getChildren().addAll(generateButton, saveBtn, closeBtn);
        return box;
    }

    private void refreshModelList() {
        List<String> models = RLModelManager.listModels();
        modelComboBox.getItems().clear();
        modelComboBox.getItems().addAll(models);
    }

    private void setControlsDisabled(boolean disabled) {
        if (mainContent != null) {
            mainContent.setDisable(disabled);
        }
    }

    private void createNewModel() {
        String name = newModelField.getText().trim();
        if (name.isEmpty()) { showAlert("Enter a model name."); return; }
        if (name.contains(" ") || name.contains("/") || name.contains("\\")) {
            showAlert("Model name cannot contain spaces or slashes."); return;
        }
        
        setControlsDisabled(true);
        setStatus("⏳ Creating model '" + name + "' (Initializing Neural Network)...", "#e67e22");

        Thread t = new Thread(() -> {
            RLTrainingService newService = new RLTrainingService(); // Heavy operation
            Platform.runLater(() -> {
                currentModelName = name;
                rlService = newService;
                // Sync service with UI state
                rlService.setUseMultiDiscreteFlow(multiDiscreteCheck.isSelected());
                
                generateButton.setDisable(true);
                setStatus("✅ Model '" + name + "' created.", "#2ecc71");
                saveCurrentModel();
                refreshModelList();
                updateStatsLabels();
                setControlsDisabled(false);
            });
        });
        t.setDaemon(true);
        t.start();
    }

    private void loadSelectedModel() {
        String name = modelComboBox.getValue();
        if (name == null || name.isEmpty()) { showAlert("Select a model to load."); return; }
        
        setControlsDisabled(true);
        setStatus("⏳ Loading model '" + name + "'...", "#e67e22");

        Thread t = new Thread(() -> {
            try {
                RLTrainingService newService = new RLTrainingService(); // Heavy operation
                RLModel model = RLModelManager.loadModel(name, newService);
                
                Platform.runLater(() -> {
                    rlService = newService;
                    currentModelName = name;
                    RLModelManager.restoreFromModel(rlService, model);
                    
                    // Sync service with UI state (UI is truth)
                    rlService.setUseMultiDiscreteFlow(multiDiscreteCheck.isSelected());

                    logisticsSlider.setValue(model.logisticsWeight);
                    costSlider.setValue(model.costWeight);
                    raidSlider.setValue(model.raidWeight);
                    updateSliderLabels();
                    generateButton.setDisable(rlService.getBestScore() <= 0);
                    setStatus(String.format("📂 Loaded '%s'", name), "#3498db");
                    updateStatsLabels();
                    setControlsDisabled(false);
                });
            } catch (Exception ex) {
                Platform.runLater(() -> {
                    showAlert("Failed to load model: " + ex.getMessage());
                    setStatus("Error loading model.", "#e74c3c");
                    setControlsDisabled(false);
                });
            }
        });
        t.setDaemon(true);
        t.start();
    }

    private void deleteSelectedModel() {
        String name = modelComboBox.getValue();
        if (name == null || name.isEmpty()) return;
        if (RLModelManager.deleteModel(name)) {
            refreshModelList();
            if (name.equals(currentModelName)) {
                currentModelName = null;
                rlService = null;
                generateButton.setDisable(true);
                setStatus("🗑 Model deleted.", "#aaa");
                updateStatsLabels();
            }
        }
    }

    private void saveCurrentModel() {
        saveCurrentModel(false);
    }

    private void saveCurrentModel(boolean silent) {
        if (currentModelName == null) { if (!silent) showAlert("No model active. Create or load a model first."); return; }
        try {
            double[] w = getNormalizedWeights();
            RLModel model = RLModelManager.createSnapshot(currentModelName, rlService, w[0], w[1], w[2]);
            RLModelManager.saveModel(model, rlService);
            if (!silent) setStatus("💾 Model '" + currentModelName + "' saved.", "#aaa");
        } catch (Exception ex) {
            if (!silent) showAlert("Failed to save model: " + ex.getMessage());
        }
    }

    private void startTraining() {
        if (currentModelName == null) { showAlert("Create or load a model first."); return; }
        if (rlService == null) { showAlert("Create or load a model first."); return; }

        int episodes = episodesSpinner.getValue();
        int maxSteps = stepsSpinner.getValue();
        int epochs = (int) Math.round(epochSlider.getValue());
        double[] w = getNormalizedWeights();
        int totalEpisodes = episodes * epochs;

        trainButton.setDisable(true);
        generateButton.setDisable(true);
        progressBar.setVisible(true);
        progressBar.setProgress(0);
        
        // Final sync before training
        rlService.setUseMultiDiscreteFlow(multiDiscreteCheck.isSelected());
        System.out.println("[RL MODE] UI checkbox=" + multiDiscreteCheck.isSelected() 
            + ", service.useMultiDiscreteFlow=" + rlService.isUseMultiDiscreteFlow());

        setStatus("⏳ Training...", "#e67e22");
        timerLabel.setText("⏱ Training time: 00:00:00");

        final double logW = w[0], costW = w[1], raidW = w[2];

        final double[] lastFoundBest = { rlService.getBestScore() };
        baseHistoryCombo.getItems().clear();

        // Initialize CSV training log
        rlService.setLogFile(currentModelName);

        // Start elapsed-time ticker
        final long startTime = System.currentTimeMillis();
        timerRunning = true;
        Thread timerThread = new Thread(() -> {
            while (timerRunning) {
                try { Thread.sleep(1000); } catch (InterruptedException ignored) { break; }
                long elapsed = (System.currentTimeMillis() - startTime) / 1000;
                long h = elapsed / 3600, m = (elapsed % 3600) / 60, s = elapsed % 60;
                String timeStr = String.format("⏱ Training time: %02d:%02d:%02d", h, m, s);
                Platform.runLater(() -> timerLabel.setText(timeStr));
            }
        });
        timerThread.setDaemon(true);
        timerThread.start();

        Thread trainThread = new Thread(() -> {
            rlService.train(episodes, maxSteps, logW, costW, raidW, epochs, metrics -> {
                int totalDone = (int)((metrics.currentEpoch - 1) * metrics.totalEpisodesPerEpoch + metrics.currentEpisodeInEpoch);

                Platform.runLater(() -> {
                    progressBar.setProgress((double) totalDone / totalEpisodes);
                    setStatus(String.format(java.util.Locale.US,
                        "Epoch %d/%d — Ep %d/%d | Best: %.3f | AvgEval: %.3f | ε: %.3f",
                        metrics.currentEpoch, metrics.totalEpochs, metrics.currentEpisodeInEpoch, metrics.totalEpisodesPerEpoch, metrics.bestScore, metrics.avgEvalScore, metrics.epsilon), "#e67e22");
                    
                    // Update extended stats directly from callback data
                    avgEvalLabel.setText(String.format(java.util.Locale.US, "📈 Avg Eval Score (last 50): %.4f", metrics.avgEvalScore));
                    avgRewardLabel.setText(String.format(java.util.Locale.US, "📊 Avg Step Reward: %.4f", metrics.avgReward));
                    invalidRateLabel.setText(String.format(java.util.Locale.US, "❌ Invalid Action Rate: %.1f%%", metrics.invalidActionRate * 100));
                    bestBaseLabel.setText(String.format("🧱 Best Base: Блоків: %d | ТЦ: %s | Дверей: %d",
                        metrics.bestBaseBlocks, metrics.bestBaseHasTC ? "Є" : "Нема", metrics.bestBaseDoors));
                    
                    long usedMB = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
                    long maxMB = Runtime.getRuntime().maxMemory() / (1024 * 1024);
                    memoryUsageLabel.setText(String.format("💾 RAM: %d MB / %d MB", usedMB, maxMB));
                    
                    if (metrics.bestScore > lastFoundBest[0] && rlService.getBestGridModel() != null) {
                        lastFoundBest[0] = metrics.bestScore;
                        BaseSnapshot snap = new BaseSnapshot(
                            String.format(java.util.Locale.US, "BEST! %.2f (Ep %d)", metrics.bestScore, totalDone), 
                            rlService.getBestGridModel());
                        baseHistoryCombo.getItems().add(snap);
                        baseHistoryCombo.getSelectionModel().select(snap);
                    }
                    
                    // Save every 5th base (modulo 5)
                    if (totalDone > 0 && totalDone % 5 == 0 && rlService.getCurrentEpisodeGrid() != null) {
                        String snapName = String.format(java.util.Locale.US, 
                            "Score: %.2f | R:%.1f F:%.1f (Ep %d)", 
                            metrics.currentEpisodeEvalScore, metrics.currentEpisodeStepReward, metrics.currentEpisodeFinalReward, totalDone);
                        BaseSnapshot snap = new BaseSnapshot(snapName, rlService.getCurrentEpisodeGrid());
                        baseHistoryCombo.getItems().add(snap);
                    }
                    
                    updateStatsLabels();
                });
            }, () -> {
                // Epoch complete callback — auto-save
                Platform.runLater(() -> {
                    saveCurrentModel(true);
                    refreshModelList();
                    // Auto-generate best on canvas after each epoch
                    if (rlService.getBestScore() > 0 && rlService.getBestGridModel() != null) {
                        gridModel.clear();
                        for (com.rustbuilder.model.core.BuildingBlock b : rlService.getBestGridModel().getAllBlocks()) {
                            gridModel.addBlockSilent(b);
                        }
                        gridModel.finalizeLoad();
                        gameCanvas.draw();
                    }
                });
            });

            Platform.runLater(() -> {
                timerRunning = false;
                trainButton.setDisable(false);
                generateButton.setDisable(rlService.getBestScore() <= 0);
                progressBar.setVisible(false);
                setStatus(String.format(java.util.Locale.US, "✅ Done — %d epochs, %d total episodes, Best: %.3f",
                    epochs, totalEpisodes, rlService.getBestScore()), "#2ecc71");
                updateStatsLabels();
                saveCurrentModel(true);
                refreshModelList();
            });
        });
        trainThread.setDaemon(true);
        trainThread.start();
    }

    private void generateBest() {
        GridModel bestGrid = rlService.getBestGridModel();
        if (bestGrid == null || bestGrid.getAllBlocks().isEmpty()) {
            showAlert("No valid generated layout. Train first.");
            return;
        }

        gridModel.clear();
        for (com.rustbuilder.model.core.BuildingBlock b : bestGrid.getAllBlocks()) {
            gridModel.addBlockSilent(b);
        }
        gridModel.finalizeLoad();
        gameCanvas.draw();
        
        setStatus(String.format("🏠 Generated best base (Score: %.2f)", rlService.getBestScore()), "#9b59b6");
    }

    private void updateStatsLabels() {
        if (rlService != null) {
            com.rustbuilder.ai.core.TrainingMetrics metrics = rlService.getMetrics();
            rewardStatsLabel.setText(String.format(java.util.Locale.US, "🏆 Best Eval Score: %.3f", metrics.bestScore));
            avgEvalLabel.setText(String.format(java.util.Locale.US, "📈 Avg Eval Score (last 50): %.4f", metrics.avgEvalScore));
            avgRewardLabel.setText(String.format(java.util.Locale.US, "📊 Avg Step Reward: %.4f", metrics.avgReward));
            
            int inv = metrics.lastEpisodeInvalidActions;
            int tot = metrics.lastEpisodeTotalActions;
            double pct = tot > 0 ? (double) inv / tot * 100 : 0;
            invalidRateLabel.setText(String.format(java.util.Locale.US, "❌ Invalid Action Rate: %.1f%% (%d/%d)", pct, inv, tot));
            
            long usedMB = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
            long maxMB = Runtime.getRuntime().maxMemory() / (1024 * 1024);
            memoryUsageLabel.setText(String.format("💾 RAM: %d MB / %d MB", usedMB, maxMB));
            
            bestBaseLabel.setText(String.format("🧱 Best Base: Блоків: %d | ТЦ: %s | Дверей: %d",
                metrics.bestBaseBlocks,
                metrics.bestBaseHasTC ? "Є" : "Нема",
                metrics.bestBaseDoors));
            
            qTableSizeLabel.setText(String.format(java.util.Locale.US, "🧠 Memory / Q-Table Size: %d", metrics.memorySize));
            episodeStatsLabel.setText(String.format(java.util.Locale.US, "🔁 Total Episodes: %d", metrics.totalEpisodesTrained));
            epsilonStatsLabel.setText(String.format(java.util.Locale.US, "🎲 Epsilon (ε): %.4f", metrics.epsilon));
            lossStatsLabel.setText(String.format(java.util.Locale.US, "📉 Loss: %.6f", metrics.lastTrainLoss));
        } else {
            rewardStatsLabel.setText("🏆 Best Eval Score: —");
            avgEvalLabel.setText("📈 Avg Eval Score (last 50): —");
            avgRewardLabel.setText("📊 Avg Step Reward: —");
            invalidRateLabel.setText("❌ Invalid Action Rate: —");
            memoryUsageLabel.setText("💾 RAM: —");
            bestBaseLabel.setText("🧱 Best Base: —");
            qTableSizeLabel.setText("🧠 Replay Memory Size: 0");
            episodeStatsLabel.setText("🔁 Total Episodes: 0");
            epsilonStatsLabel.setText("🎲 Epsilon (ε): —");
            lossStatsLabel.setText("📉 Loss: —");
        }
    }

    public void show() { dialogStage.showAndWait(); }

    private VBox card() {
        VBox v = new VBox(7);
        v.setStyle(CARD_STYLE);
        return v;
    }

    private Label sectionTitle(String text) {
        Label l = new Label(text);
        l.setStyle(SECTION_TITLE);
        return l;
    }

    private Label bodyLabel(String text) {
        Label l = new Label(text);
        l.setStyle(BODY_LABEL);
        return l;
    }

    private Label valueLabel(String text) {
        Label l = new Label(text);
        l.setStyle(VALUE_LABEL);
        return l;
    }

    private Button styledBtn(String text, String color) {
        Button b = new Button(text);
        b.setStyle("-fx-background-color: " + color + "; -fx-text-fill: white;");
        return b;
    }

    private Slider buildSlider(double min, double max, double def, double tickUnit) {
        Slider s = new Slider(min, max, def);
        s.setShowTickLabels(true);
        s.setShowTickMarks(true);
        s.setMajorTickUnit(tickUnit);
        return s;
    }

    private void setStatus(String text, String color) {
        statusLabel.setText(text);
        statusLabel.setStyle("-fx-text-fill: " + color + "; -fx-font-size: 11px;");
    }

    private void showAlert(String msg) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("RL AI Generator");
        alert.setHeaderText(null);
        alert.setContentText(msg);
        alert.showAndWait();
    }
}
