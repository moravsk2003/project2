package com.rustbuilder.ai.ea;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.deployable.LootRoom;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.TriangleFloor;
import com.rustbuilder.model.structure.TriangleFoundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.model.deployable.Workbench;

/**
 * Represents a house design as a list of building actions.
 * This is the "DNA" that the genetic algorithm evolves.
 */
public class BaseGenome implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final Random RNG = new Random();

    /** Maximum grid dimension (NxN tiles) for random placement */
    public static final int MAX_GRID = 8;
    /** Maximum floors */
    public static final int MAX_FLOORS = 8;
    /** Min/max actions in a genome */
    private static final int MIN_ACTIONS = 10;
    private static final int MAX_ACTIONS = 60;

    private List<BuildAction> actions;
    private double fitness = -1;

    /**
     * A single build instruction.
     */
    public static class BuildAction implements Serializable {
        private static final long serialVersionUID = 1L;

        public enum ActionType {
            FOUNDATION, TRIANGLE_FOUNDATION,
            WALL, DOORWAY, WINDOW_FRAME, DOOR,
            FLOOR, TRIANGLE_FLOOR,
            TC, WORKBENCH, LOOT_ROOM
        }

        public final ActionType actionType;
        public final int gridX;     // tile x (0..MAX_GRID-1)
        public final int gridY;     // tile y (0..MAX_GRID-1)
        public final int floor;     // z level (0..MAX_FLOORS-1)
        public final int orientation; // 0=N, 1=E, 2=S, 3=W
        public final int tier;      // 0=TWIG..4=HQM
        public final int doorType;  // 0=SHEET_METAL, 1=GARAGE, 2=ARMORED

        public BuildAction(ActionType actionType, int gridX, int gridY, int floor,
                           int orientation, int tier, int doorType) {
            this.actionType = actionType;
            this.gridX = gridX;
            this.gridY = gridY;
            this.floor = floor;
            this.orientation = orientation;
            this.tier = tier;
            this.doorType = doorType;
        }

        public static BuildAction random() {
            ActionType type = getRandomActionType();
            
            // Bias coordinates towards the center (using Gaussian) to ensure blocks connect
            int gx = (int) Math.round(3.5 + RNG.nextGaussian() * 2.0);
            int gy = (int) Math.round(3.5 + RNG.nextGaussian() * 2.0);
            gx = Math.max(0, Math.min(MAX_GRID - 1, gx));
            gy = Math.max(0, Math.min(MAX_GRID - 1, gy));

            return new BuildAction(
                type,
                gx, gy,
                RNG.nextInt(MAX_FLOORS),
                RNG.nextInt(4),
                2 + RNG.nextInt(3), // tier: STONE(2), METAL(3), HQM(4)
                RNG.nextInt(3)      // door type
            );
        }

        public BuildAction mutated() {
            // Mutate one random field
            int field = RNG.nextInt(7);
            
            ActionType newType = actionType;
            int newX = gridX;
            int newY = gridY;
            int newFloor = floor;
            int newOrient = orientation;
            int newTier = tier;
            int newDoor = doorType;

            switch (field) {
                case 0: newType = getRandomActionType(); break;
                case 1: newX = Math.max(0, Math.min(MAX_GRID - 1, gridX + (int)Math.round(RNG.nextGaussian()))); break;
                case 2: newY = Math.max(0, Math.min(MAX_GRID - 1, gridY + (int)Math.round(RNG.nextGaussian()))); break;
                case 3: newFloor = Math.max(0, Math.min(MAX_FLOORS - 1, floor + (RNG.nextBoolean() ? 1 : -1))); break;
                case 4: newOrient = (orientation + (RNG.nextBoolean() ? 1 : -1) + 4) % 4; break;
                case 5: newTier = 2 + RNG.nextInt(3); break; // Random stone/metal/hqm
                case 6: newDoor = RNG.nextInt(3); break;
            }

            return new BuildAction(newType, newX, newY, newFloor, newOrient, newTier, newDoor);
        }

        private static ActionType getRandomActionType() {
            // Weighted random selection.
            // TC chance raised from 4% → 10%: logistics score is 0 without TC,
            // so the AI needs TC to appear frequently to get meaningful fitness signal.
            double r = RNG.nextDouble();
            if (r < 0.20) return ActionType.FOUNDATION;          // 20%
            if (r < 0.37) return ActionType.WALL;                // 17%
            if (r < 0.52) return ActionType.FLOOR;               // 15%
            if (r < 0.59) return ActionType.TRIANGLE_FOUNDATION; //  7%
            if (r < 0.66) return ActionType.TRIANGLE_FLOOR;      //  7%
            if (r < 0.78) return ActionType.DOORWAY;             // 12%
            if (r < 0.83) return ActionType.WINDOW_FRAME;        //  5%
            if (r < 0.84) return ActionType.DOOR;                //  1%
            if (r < 0.94) return ActionType.TC;                  // 10%
            if (r < 0.97) return ActionType.LOOT_ROOM;           //  3%
            return ActionType.WORKBENCH;                         //  3%
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(actionType, gridX, gridY, floor, orientation, tier, doorType);
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            BuildAction other = (BuildAction) obj;
            return actionType == other.actionType &&
                   gridX == other.gridX &&
                   gridY == other.gridY &&
                   floor == other.floor &&
                   orientation == other.orientation &&
                   tier == other.tier &&
                   doorType == other.doorType;
        }
    }

    public BaseGenome() {
        this.actions = new ArrayList<>();
    }

    public BaseGenome(List<BuildAction> actions) {
        this.actions = new ArrayList<>(actions);
    }

    /**
     * Create a random genome with a random number of actions.
     */
    public static BaseGenome randomGenome() {
        BaseGenome genome = new BaseGenome();
        int count = MIN_ACTIONS + RNG.nextInt(MAX_ACTIONS - MIN_ACTIONS);
        for (int i = 0; i < count; i++) {
            genome.actions.add(BuildAction.random());
        }
        return genome;
    }

    /**
     * Like {@link #randomGenome()} but guarantees at least one TC action placed
     * near the grid centre. Without a TC the logistics score is always 0, which
     * makes the fitness landscape totally flat and stops evolution in its tracks.
     */
    public static BaseGenome randomGenomeWithTC() {
        BaseGenome genome = randomGenome();
        boolean hasTC = genome.actions.stream()
                .anyMatch(a -> a.actionType == BuildAction.ActionType.TC);
        if (!hasTC) {
            genome.actions.add(new BuildAction(
                BuildAction.ActionType.TC,
                3 + RNG.nextInt(2), 3 + RNG.nextInt(2), 0, 0, 2, 0
            ));
        }
        return genome;
    }

    /**
     * Single-point crossover between two parents.
     */
    public static BaseGenome crossover(BaseGenome parent1, BaseGenome parent2) {
        List<BuildAction> a1 = parent1.actions;
        List<BuildAction> a2 = parent2.actions;

        // Fix: guarantee at least 1 gene from each parent so crossover
        // actually transfers genetic material (not just a random genome).
        int cut1 = 1 + RNG.nextInt(Math.max(1, a1.size() - 1)); // 1..size-1 from parent1
        int cut2 = RNG.nextInt(Math.max(1, a2.size()));          // 0..size-1 tail from parent2

        List<BuildAction> child = new ArrayList<>();
        // First part from parent1
        for (int i = 0; i < cut1 && i < a1.size(); i++) {
            child.add(a1.get(i));
        }
        // Second part from parent2
        for (int i = cut2; i < a2.size(); i++) {
            child.add(a2.get(i));
        }

        // Clamp size
        if (child.size() > MAX_ACTIONS) {
            child = new ArrayList<>(child.subList(0, MAX_ACTIONS));
        }
        if (child.size() < MIN_ACTIONS) {
            while (child.size() < MIN_ACTIONS) {
                child.add(BuildAction.random());
            }
        }

        return new BaseGenome(child);
    }

    /**
     * Mutate the genome — modify, add, or remove random actions.
     */
    public void mutate(double mutationRate) {
        // Mutate existing actions
        for (int i = 0; i < actions.size(); i++) {
            if (RNG.nextDouble() < mutationRate) {
                actions.set(i, actions.get(i).mutated());
            }
        }
        // Possibly add a new action
        if (RNG.nextDouble() < mutationRate && actions.size() < MAX_ACTIONS) {
            actions.add(RNG.nextInt(actions.size() + 1), BuildAction.random());
        }
        // Possibly remove an action
        if (RNG.nextDouble() < mutationRate && actions.size() > MIN_ACTIONS) {
            actions.remove(RNG.nextInt(actions.size()));
        }
        // Fix: if the genome has no TC at all, add one with 30% chance.
        // Without TC, logisticsScore = 0 → fitness ≈ 0 → genome is dead weight.
        boolean hasTC = actions.stream().anyMatch(a -> a.actionType == BuildAction.ActionType.TC);
        if (!hasTC && RNG.nextDouble() < 0.30) {
            // Place TC near the center of the grid so it has a chance to snap
            BuildAction tcAction = new BuildAction(
                BuildAction.ActionType.TC,
                3 + RNG.nextInt(2), 3 + RNG.nextInt(2), 0, 0, 2, 0
            );
            actions.add(tcAction);
        }
    }

    /**
     * Decode this genome into building blocks on the given GridModel.
     * Returns true if at least some blocks were placed.
     */
    public boolean decode(GridModel gridModel) {
        gridModel.clear();

        com.rustbuilder.service.physics.SnappingService snappingService = new com.rustbuilder.service.physics.SnappingService(gridModel);

        double tileSize = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double halfTile = com.rustbuilder.config.GameConstants.HALF_TILE;
        double startX = 200;
        double startY = 200;

        int placed = 0;

        // ── Ordered placement: 3 phases ──────────────────────────────────────
        // Phase order ensures that when a wall/floor tries to snap, foundations
        // already exist in the grid. Furniture is placed last so it lands on a
        // built floor rather than floating in the air.

        for (int phase = 0; phase < 3; phase++) {
            for (BuildAction action : actions) {

                // Phase gate
                boolean isFoundation = action.actionType == BuildAction.ActionType.FOUNDATION
                                    || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION;
                boolean isFurniture  = action.actionType == BuildAction.ActionType.TC
                                    || action.actionType == BuildAction.ActionType.WORKBENCH
                                    || action.actionType == BuildAction.ActionType.LOOT_ROOM;
                boolean isStructural = !isFoundation && !isFurniture; // walls, floors, doors

                if (phase == 0 && !isFoundation) continue;
                if (phase == 1 && !isStructural) continue;
                if (phase == 2 && !isFurniture)  continue;

                com.rustbuilder.util.GridPlacementUtils.Placement placement = com.rustbuilder.util.GridPlacementUtils.calculatePlacement(gridModel, action);

                if (!placement.valid) {
                    continue; // Must successfully snap to existing structures
                }

                double finalX           = placement.x;
                double finalY           = placement.y;
                double finalRotation    = placement.rotation;
                Orientation finalOrientation = placement.orientation;
                int z = isFoundation ? 0 : action.floor;

                // ── Build block ──────────────────────────────────────────────
                BuildingTier tier     = tierFromInt(action.tier);
                DoorType     doorType = doorTypeFromInt(action.doorType);
                BuildingBlock block   = null;

                switch (action.actionType) {
                    case FOUNDATION:
                        block = new Foundation(finalX, finalY, z);
                        block.setRotation(finalRotation);
                        break;
                    case TRIANGLE_FOUNDATION:
                        block = new TriangleFoundation(finalX, finalY, z, finalRotation);
                        break;
                    case WALL:
                        block = new Wall(finalX, finalY, z, finalOrientation);
                        break;
                    case DOORWAY:
                        Wall dw = new Wall(finalX, finalY, z, finalOrientation);
                        dw.setType(BuildingType.DOORWAY);
                        dw.setDoorType(doorType);
                        block = dw;
                        break;
                    case WINDOW_FRAME:
                        Wall wf = new Wall(finalX, finalY, z, finalOrientation);
                        wf.setType(BuildingType.WINDOW_FRAME);
                        block = wf;
                        break;
                    case DOOR:
                        block = new Door(finalX, finalY, z, finalOrientation, doorType);
                        break;
                    case FLOOR:
                        block = new Floor(finalX, finalY, z, finalRotation);
                        break;
                    case TRIANGLE_FLOOR:
                        block = new TriangleFloor(finalX, finalY, z, finalRotation);
                        break;
                    case TC:
                        block = new ToolCupboard(finalX, finalY, z, finalRotation);
                        break;
                    case WORKBENCH:
                        block = new Workbench(finalX, finalY, z, finalRotation);
                        break;
                    case LOOT_ROOM:
                        block = new LootRoom(finalX, finalY, z, finalRotation);
                        break;
                }

                if (block != null) {
                    // Snap rotation is authoritative for non-wall blocks
                    if (!(block instanceof Wall)) {
                        block.setRotation(finalRotation);
                    }
                    block.setTier(tier);

                    if (isFurniture || isFoundation) {
                        if (gridModel.canPlace(block)) {
                            gridModel.addBlockSilent(block);
                            placed++;
                        }
                    } else {
                        if (!gridModel.hasCollision(block)) {
                            if (gridModel.addBlockSilent(block)) {
                                placed++;
                            }
                        }
                    }
                }
            }
        }

        // Single stability pass: propagates support from foundations up,
        // then evicts any blocks that ended up unsupported.
        gridModel.finalizeLoad();

        return placed > 0;
    }


    private String actionTypeToToolName(BuildAction.ActionType type) {
        switch (type) {
            case FOUNDATION: return "FOUNDATION";
            case TRIANGLE_FOUNDATION: return "TRIANGLE";
            case WALL: return "WALL";
            case DOORWAY: return "DOOR_FRAME";
            case WINDOW_FRAME: return "WINDOW_FRAME";
            case DOOR: return "DOOR";
            case FLOOR: return "FLOOR";
            case TRIANGLE_FLOOR: return "TRIANGLE_FLOOR";
            case TC: return "TC";
            case WORKBENCH: return "WORKBENCH";
            case LOOT_ROOM: return "LOOT_ROOM";
            default: return "WALL";
        }
    }

    // --- Helpers ---

    private static BuildingTier tierFromInt(int t) {
        switch (t) {
            case 0: return BuildingTier.TWIG;
            case 1: return BuildingTier.WOOD;
            case 2: return BuildingTier.STONE;
            case 3: return BuildingTier.METAL;
            case 4: return BuildingTier.HQM;
            default: return BuildingTier.STONE;
        }
    }

    private static DoorType doorTypeFromInt(int d) {
        switch (d) {
            case 0: return DoorType.SHEET_METAL;
            case 1: return DoorType.GARAGE;
            case 2: return DoorType.ARMORED;
            default: return DoorType.SHEET_METAL;
        }
    }

    private static Orientation orientFromInt(int o) {
        switch (o) {
            case 0: return Orientation.NORTH;
            case 1: return Orientation.EAST;
            case 2: return Orientation.SOUTH;
            case 3: return Orientation.WEST;
            default: return Orientation.NORTH;
        }
    }

    // --- Getters / Setters ---

    public List<BuildAction> getActions() {
        return actions;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public BaseGenome copy() {
        BaseGenome clone = new BaseGenome(new ArrayList<>(this.actions));
        clone.fitness = this.fitness;
        return clone;
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(actions);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        BaseGenome other = (BaseGenome) obj;
        return java.util.Objects.equals(actions, other.actions);
    }
}
