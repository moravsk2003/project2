package com.rustbuilder.ai.rl.multidiscrete;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.rustbuilder.ai.rl.legacy.ActionSpace;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.util.GridPlacementUtils;

/**
 * Provides deep conditional masking to prune invalid actions during multi-discrete selection.
 * Uses real physics simulation (dry-runs) to ensure selected components are mathematically possible.
 */
public class HeuristicMaskingUtils {

    private static final int GRID_SIZE = MultiDiscreteActionSpace.GRID_SIZE;
    private static final int MAX_FLOORS = MultiDiscreteActionSpace.FLOOR_COUNT;

    public static boolean DEBUG_MODE = false;

    // Debug counters for dead branches
    public static int prunedTypeCount = 0;
    public static int emptyTilesCount = 0;
    public static int emptyRotationsCount = 0;

    /**
     * Phase 1: Get globally valid building types for the current step.
     */
    public static List<Integer> getValidTypes(GridModel grid, boolean hasTC, boolean hasLootRoom, int step) {
        return ActionSpace.getValidTypeActions(hasTC, hasLootRoom, step);
    }

    /**
     * Phase 1 (Feasible): Get types that have AT LEAST ONE valid full continuation (floor -> tile -> rot).
     */
    public static List<Integer> getFeasibleTypes(GridModel grid, boolean hasTC, boolean hasLootRoom, int step) {
        List<Integer> initialTypes = getValidTypes(grid, hasTC, hasLootRoom, step);
        List<Integer> feasibleTypes = new ArrayList<>();

        for (int typeIdx : initialTypes) {
            if (hasFeasibleContinuation(grid, typeIdx, step)) {
                feasibleTypes.add(typeIdx);
            } else {
                prunedTypeCount++;
            }
        }
        return feasibleTypes;
    }

    private static boolean hasFeasibleContinuation(GridModel grid, int typeIndex, int step) {
        List<Integer> possibleFloors = getBasicFloors(grid, typeIndex);
        for (int floorIndex : possibleFloors) {
            if (hasFeasibleTile(grid, typeIndex, floorIndex, step)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasFeasibleTile(GridModel grid, int typeIndex, int floorIndex, int step) {
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return true; // STOP action is always feasible

        if (step == 0) {
            int[] centerTiles = {
                (GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2 - 1),
                (GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2),
                (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2 - 1),
                (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2)
            };
            for (int tIdx : centerTiles) {
                if (hasFeasibleRotation(grid, typeIndex, floorIndex, tIdx)) {
                    return true;
                }
            }
            return false;
        }

        for (int tileIdx = 0; tileIdx < MultiDiscreteActionSpace.TILE_COUNT; tileIdx++) {
            int tx = tileIdx / GRID_SIZE;
            int ty = tileIdx % GRID_SIZE;

            if (type != com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.FOUNDATION && type != com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TRIANGLE_FOUNDATION) {
                List<BuildingBlock> near = grid.getNearbyBlocks(200.0 + tx * GameConstants.TILE_SIZE, 200.0 + ty * GameConstants.TILE_SIZE, floorIndex, 120.0);
                if (near.isEmpty()) continue;
            }

            if (hasFeasibleRotation(grid, typeIndex, floorIndex, tileIdx)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasFeasibleRotation(GridModel grid, int typeIndex, int floorIndex, int tileIndex) {
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return true; // STOP action

        int tx = tileIndex / GRID_SIZE;
        int ty = tileIndex % GRID_SIZE;

        for (int rot = 0; rot < MultiDiscreteActionSpace.ROTATION_COUNT; rot++) {
            com.rustbuilder.ai.ea.BaseGenome.BuildAction trial = 
                new com.rustbuilder.ai.ea.BaseGenome.BuildAction(type, tx, ty, floorIndex, rot, 2, 0);
            
            GridPlacementUtils.Placement placement = GridPlacementUtils.calculatePlacement(grid, trial);
            if (placement.valid && simulateCanPlace(grid, trial, placement)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Phase 2: Get valid floors for the selected type that have continuations.
     */
    public static List<Integer> getValidFloors(GridModel grid, int typeIndex, int step) {
        List<Integer> basicFloors = getBasicFloors(grid, typeIndex);
        List<Integer> valid = new ArrayList<>();

        for (int f : basicFloors) {
            if (hasFeasibleTile(grid, typeIndex, f, step)) {
                valid.add(f);
            }
        }
        
        return valid;
    }

    private static List<Integer> getBasicFloors(GridModel grid, int typeIndex) {
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(0); // STOP action

        if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.FOUNDATION ||
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TRIANGLE_FOUNDATION) {
            return Collections.singletonList(0);
        }

        List<Integer> valid = new ArrayList<>();
        valid.add(0); // Foundations can be replaced or added at 0

        boolean[] hasSupportAtFloor = new boolean[MAX_FLOORS];
        for (BuildingBlock b : grid.getAllBlocks()) {
            if (b.getZ() >= 0 && b.getZ() < MAX_FLOORS) {
                hasSupportAtFloor[b.getZ()] = true;
            }
        }

        for (int f = 1; f < MAX_FLOORS; f++) {
            if (hasSupportAtFloor[f - 1]) {
                valid.add(f);
            }
        }
        return valid;
    }

    /**
     * Phase 3: Get valid tiles where the selected type and floor can technically be placed.
     */
    public static List<Integer> getValidTiles(GridModel grid, int typeIndex, int floorIndex, int step) {
        List<Integer> valid = new ArrayList<>();
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(0); // STOP

        if (step == 0) {
            int[] centerTiles = {
                (GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2 - 1),
                (GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2),
                (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2 - 1),
                (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2)
            };
            for (int tIdx : centerTiles) {
                if (hasFeasibleRotation(grid, typeIndex, floorIndex, tIdx)) {
                    valid.add(tIdx);
                }
            }
        } else {
            for (int tileIdx = 0; tileIdx < MultiDiscreteActionSpace.TILE_COUNT; tileIdx++) {
                int tx = tileIdx / GRID_SIZE;
                int ty = tileIdx % GRID_SIZE;

                if (type != com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.FOUNDATION && type != com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TRIANGLE_FOUNDATION) {
                    List<BuildingBlock> near = grid.getNearbyBlocks(200.0 + tx * GameConstants.TILE_SIZE, 200.0 + ty * GameConstants.TILE_SIZE, floorIndex, 120.0);
                    if (near.isEmpty()) continue;
                }

                if (hasFeasibleRotation(grid, typeIndex, floorIndex, tileIdx)) {
                    valid.add(tileIdx);
                }
            }
        }

        if (valid.isEmpty()) {
            emptyTilesCount++;
            System.err.println(String.format("[WARNING] Dead branch reached for Type: %s, Floor: %d. This should have been pruned by getFeasibleTypes!", type, floorIndex));
            if (DEBUG_MODE) {
                valid.add(MultiDiscreteActionSpace.TILE_COUNT / 2);
            } else {
                throw new IllegalStateException("Dead branch: no valid tiles for selected type and floor.");
            }
        }

        return valid;
    }

    /**
     * Phase 4: Get valid rotations (0..3) for the selected type, floor, and tile.
     */
    public static List<Integer> getValidRotations(GridModel grid, int typeIndex, int floorIndex, int tileIndex) {
        List<Integer> valid = new ArrayList<>();
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(0); // STOP

        int tx = tileIndex / GRID_SIZE;
        int ty = tileIndex % GRID_SIZE;

        for (int rot = 0; rot < MultiDiscreteActionSpace.ROTATION_COUNT; rot++) {
            com.rustbuilder.ai.ea.BaseGenome.BuildAction trial = 
                new com.rustbuilder.ai.ea.BaseGenome.BuildAction(type, tx, ty, floorIndex, rot, 2, 0);
            
            GridPlacementUtils.Placement placement = GridPlacementUtils.calculatePlacement(grid, trial);
            if (placement.valid && simulateCanPlace(grid, trial, placement)) {
                valid.add(rot);
            }
        }

        if (valid.isEmpty()) {
            emptyRotationsCount++;
            System.err.println(String.format("[WARNING] Dead branch reached for Type: %s, Floor: %d, Tile: %d. This should have been pruned by getFeasibleTypes!", type, floorIndex, tileIndex));
            if (DEBUG_MODE) {
                valid.add(0);
            } else {
                throw new IllegalStateException("Dead branch: no valid rotations for selected type, floor, and tile.");
            }
        }

        return valid;
    }

    /**
     * Phase 5: Frozen aim sector mapping.
     */
    public static List<Integer> getValidAimSectors() {
        return Collections.singletonList(12); // Center fixed
    }

    /**
     * DRY-RUN simulation helper.
     */
    private static boolean simulateCanPlace(GridModel grid, com.rustbuilder.ai.ea.BaseGenome.BuildAction action, GridPlacementUtils.Placement placement) {
        double finalX = placement.x;
        double finalY = placement.y;
        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = action.floor;
        
        BuildingBlock block = null;
        switch (action.actionType) {
            case FOUNDATION: block = new Foundation(finalX, finalY, z); block.setRotation(finalRotation); break;
            case TRIANGLE_FOUNDATION: block = new TriangleFoundation(finalX, finalY, z, finalRotation); break;
            case WALL: block = new Wall(finalX, finalY, z, finalOrientation); break;
            case DOORWAY:
                Wall dw = new Wall(finalX, finalY, z, finalOrientation);
                dw.setType(BuildingType.DOORWAY);
                dw.setDoorType(DoorType.SHEET_METAL);
                block = dw;
                break;
            case WINDOW_FRAME:
                Wall wf = new Wall(finalX, finalY, z, finalOrientation);
                wf.setType(BuildingType.WINDOW_FRAME);
                block = wf;
                break;
            case DOOR: block = new Door(finalX, finalY, z, finalOrientation, DoorType.SHEET_METAL); break;
            case FLOOR: block = new Floor(finalX, finalY, z, finalRotation); break;
            case TRIANGLE_FLOOR: block = new TriangleFloor(finalX, finalY, z, finalRotation); break;
            case TC: block = new ToolCupboard(finalX, finalY, z, finalRotation); break;
            case WORKBENCH: block = new Workbench(finalX, finalY, z, finalRotation); break;
            case LOOT_ROOM: block = new LootRoom(finalX, finalY, z, finalRotation); break;
        }

        if (block != null) {
            block.setTier(BuildingTier.STONE);
            if (!(block instanceof Wall)) {
                block.setRotation(finalRotation);
            }

            boolean isFurniture = action.actionType == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TC || 
                                  action.actionType == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WORKBENCH || 
                                  action.actionType == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.LOOT_ROOM;

            if (isFurniture) {
                if (grid.canPlace(block)) {
                    boolean occupied = false;
                    for (BuildingBlock b2 : grid.getAllBlocks()) {
                        if (com.rustbuilder.util.BuildingTypeUtils.isFurniture(b2.getType()) &&
                            b2.getZ() == block.getZ() &&
                            Math.abs(b2.getX() - block.getX()) < 1.0 &&
                            Math.abs(b2.getY() - block.getY()) < 1.0) {
                            occupied = true;
                            break;
                        }
                    }
                    return !occupied;
                }
                return false;
            } else {
                return grid.canPlace(block);
            }
        }
        return false;
    }
}

