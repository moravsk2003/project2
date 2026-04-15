package com.rustbuilder;

import com.rustbuilder.service.generator.GeneratorService;
import com.rustbuilder.model.core.*;
import com.rustbuilder.service.physics.*;
import com.rustbuilder.service.economy.*;

import com.rustbuilder.controller.GameController;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.ui.GameCanvas;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Separator;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class MainApp extends Application {

    private GameController gameController;

    @Override
    public void start(Stage stage) {
        try {
            GridModel gridModel = new GridModel();
            GameCanvas gameCanvas = new GameCanvas(gridModel, 800, 600);
            gameController = new GameController(gridModel, gameCanvas);
            gameCanvas.setController(gameController);

            // === Building Tools ===
            ToolBar toolBar = new ToolBar();

            Button btnFoundation = new Button("Foundation");
            btnFoundation.setOnAction(e -> gameController.setSelectedTool("FOUNDATION"));

            Button btnTriFoundation = new Button("Tri Foundation");
            btnTriFoundation.setOnAction(e -> gameController.setSelectedTool("TRIANGLE"));

            Button btnWall = new Button("Wall");
            btnWall.setOnAction(e -> gameController.setSelectedTool("WALL"));

            Button btnFloor = new Button("Floor");
            btnFloor.setOnAction(e -> gameController.setSelectedTool("FLOOR"));

            Button btnTriFloor = new Button("Tri Floor");
            btnTriFloor.setOnAction(e -> gameController.setSelectedTool("TRIANGLE_FLOOR"));

            Button btnDoorFrame = new Button("Door Frame");
            btnDoorFrame.setOnAction(e -> gameController.setSelectedTool("DOOR_FRAME"));

            Button btnWindowFrame = new Button("Window");
            btnWindowFrame.setOnAction(e -> gameController.setSelectedTool("WINDOW_FRAME"));

            Button btnDoor = new Button("Door");
            btnDoor.setOnAction(e -> gameController.setSelectedTool("DOOR"));

            Button btnToolCupboard = new Button("TC");
            btnToolCupboard.setOnAction(e -> gameController.setSelectedTool("TC"));

            Button btnWorkbench = new Button("Workbench");
            btnWorkbench.setOnAction(e -> gameController.setSelectedTool("WORKBENCH"));

            Button btnLootRoom = new Button("Loot Room");
            btnLootRoom.setOnAction(e -> gameController.setSelectedTool("LOOT_ROOM"));

            // === Navigation ===
            Button btnUp = new Button("▲");
            btnUp.setOnAction(e -> gameController.moveFloorUp());

            Button btnDown = new Button("▼");
            btnDown.setOnAction(e -> gameController.moveFloorDown());

            Button btnDelete = new Button("Delete");
            btnDelete.setOnAction(e -> gameController.setSelectedTool("DELETE"));

            Button btnClear = new Button("Clear");
            btnClear.setOnAction(e -> gameController.clearGrid());

            // === Tier Selection ===
            ComboBox<String> tierBox = new ComboBox<>();
            tierBox.getItems().addAll("Twig", "Wood", "Stone", "Metal", "HQM");
            tierBox.setValue("Stone");
            tierBox.setOnAction(e -> {
                String sel = tierBox.getValue();
                switch (sel) {
                    case "Twig":  gameController.setSelectedTier(BuildingTier.TWIG); break;
                    case "Wood":  gameController.setSelectedTier(BuildingTier.WOOD); break;
                    case "Stone": gameController.setSelectedTier(BuildingTier.STONE); break;
                    case "Metal": gameController.setSelectedTier(BuildingTier.METAL); break;
                    case "HQM":   gameController.setSelectedTier(BuildingTier.HQM); break;
                }
            });

            // === Door Type Selection ===
            ComboBox<String> doorBox = new ComboBox<>();
            doorBox.getItems().addAll("Sheet Metal", "Garage", "Armored");
            doorBox.setValue("Sheet Metal");
            doorBox.setOnAction(e -> {
                String sel = doorBox.getValue();
                switch (sel) {
                    case "Sheet Metal": gameController.setSelectedDoorType(DoorType.SHEET_METAL); break;
                    case "Garage":      gameController.setSelectedDoorType(DoorType.GARAGE); break;
                    case "Armored":     gameController.setSelectedDoorType(DoorType.ARMORED); break;
                }
            });

            // === Evaluate Button ===
            Button btnEvaluate = new Button("⚡ Evaluate");
            btnEvaluate.setStyle("-fx-background-color: #4CAF50; -fx-text-fill: white; -fx-font-weight: bold;");
            btnEvaluate.setOnAction(e -> {
                HouseEvaluator.EvaluationResult result = gameController.evaluateHouse();
                
                Alert alert = new Alert(Alert.AlertType.INFORMATION);
                alert.setTitle("House Evaluation");
                alert.setHeaderText(String.format("Final Score: %.2f / 1.00", result.finalScore));
                alert.setContentText(result.toString());
                alert.getDialogPane().setMinWidth(500);
                alert.showAndWait();
            });

            // === AI Generate Button ===
            Button btnAI = new Button("🧬 GA Generate");
            btnAI.setStyle("-fx-background-color: #2980b9; -fx-text-fill: white; -fx-font-weight: bold;");
            btnAI.setOnAction(e -> {
                com.rustbuilder.ui.GeneratorDialog dialog = 
                    new com.rustbuilder.ui.GeneratorDialog(gridModel, gameCanvas, stage);
                dialog.show();
                gameCanvas.draw(); // refresh canvas after dialog closes
            });

            // === RL AI Button ===
            Button btnRL = new Button("🤖 RL Generate");
            btnRL.setStyle("-fx-background-color: #8e44ad; -fx-text-fill: white; -fx-font-weight: bold;");
            btnRL.setOnAction(e -> {
                com.rustbuilder.ui.RLGeneratorDialog dialog = 
                    new com.rustbuilder.ui.RLGeneratorDialog(gridModel, gameCanvas, stage);
                dialog.show();
                gameCanvas.draw();
            });

            // === Stability Tool ===
            Button btnStability = new Button("🛡 Stability");
            btnStability.setOnAction(e -> {
                boolean newState = btnStability.getUserData() == null ? true : !(boolean)btnStability.getUserData();
                btnStability.setUserData(newState);

                gameCanvas.setShowStabilityTool(newState);
                if (newState) {
                    btnStability.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-font-weight: bold;");
                    gameCanvas.setGhost(0, 0, 0, null, false);
                    gameCanvas.draw();
                } else {
                    btnStability.setStyle("");
                    gameCanvas.setHoveredBlock(null);
                }
            });


            toolBar.getItems().addAll(
                    btnFoundation, btnTriFoundation, btnWall, btnFloor, btnTriFloor,
                    btnDoorFrame, btnDoor, btnWindowFrame,
                    new Separator(),
                    btnToolCupboard, btnWorkbench, btnLootRoom,
                    new Separator(),
                    btnUp, btnDown, btnDelete, btnClear,
                    new Separator(),
                    tierBox, doorBox,
                    new Separator(),
                    btnEvaluate, btnAI, btnRL,
                    new Separator(),
                    btnStability
            );

            // Mouse Events
            gameCanvas.setOnMouseMoved(e -> {
                boolean stabilityActive = btnStability.getUserData() != null && (boolean)btnStability.getUserData();
                if (stabilityActive) {
                    BuildingBlock b = gameCanvas.hitTest(e.getX(), e.getY());
                    gameCanvas.setHoveredBlock(b);
                } else {
                    gameController.handleMouseMove(e.getX(), e.getY());
                }
            });
            gameCanvas.setOnMouseClicked(e -> {
                boolean stabilityActive = btnStability.getUserData() != null && (boolean)btnStability.getUserData();
                if (!stabilityActive) {
                    gameController.handleMouseClick(e.getX(), e.getY(),
                        e.getButton() == javafx.scene.input.MouseButton.PRIMARY);
                }
            });
            
            gameCanvas.setOnScroll(e -> gameController.handleScroll(e.getDeltaY(), e.getX(), e.getY()));
            
            gameCanvas.setOnMousePressed(e -> gameController.handleMousePressed(e.getX(), e.getY(), 
                    e.getButton() == javafx.scene.input.MouseButton.MIDDLE));
            
            gameCanvas.setOnMouseDragged(e -> gameController.handleMouseDragged(e.getX(), e.getY(), 
                    e.getButton() == javafx.scene.input.MouseButton.MIDDLE));

            BorderPane root = new BorderPane();
            root.setTop(toolBar);
            root.setCenter(gameCanvas);

            // Bind canvas to container so it fills the window and redraws on resize
            gameCanvas.widthProperty().bind(root.widthProperty());
            gameCanvas.heightProperty().bind(root.heightProperty());
            gameCanvas.widthProperty().addListener(o -> gameCanvas.draw());
            gameCanvas.heightProperty().addListener(o -> gameCanvas.draw());

            Scene scene = new Scene(root, 1000, 700);
            stage.setTitle("Rust Base Builder");
            stage.setScene(scene);
            stage.show();
            gameCanvas.draw(); // draw after show() so getWidth()/getHeight() are correct
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Startup Error");
            alert.setHeaderText("Failed to initialize application");
            alert.setContentText(e.getMessage() != null ? e.getMessage() : e.toString());
            alert.showAndWait();
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
