package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.rl.*;
import com.rustbuilder.ai.rl.legacy.*;
import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.config.GameConstants;

/**
 * [RL REDESIGN]
 * A Q-Learning based decision provider for the multi-discrete phase policy.
 * 
 * NOTE: Current implementation is PLACEMENT-ONLY for the experimental/learning path.
 * Tabular provider currently functions as a 5-phase provider inside the 5-phase contract
 * (aimSector is hardcoded to center=12 and not learned) to prevent state space explosion.
 * Stop semantics for multi-discrete mode are not yet formally defined, so the 
 * STOP action is excluded from candidate generation to prevent premature rollout termination 
 * and ensure the training loop only considers constructive actions.
 */
public class QTableMultiDiscreteDecisionProvider implements MultiDiscretePhaseDecisionProvider {

    // Phase dimensions for the tabular space
    private static final int TYPES = MultiDiscreteActionSpace.TYPE_COUNT; // 12
    private static final int FLOORS = MultiDiscreteActionSpace.FLOOR_COUNT; // 8
    private static final int TILES = MultiDiscreteActionSpace.TILE_COUNT; // 64
    private static final int ROTATIONS = MultiDiscreteActionSpace.ROTATION_COUNT; // 4
    
    // Total encoded actions: 12 * 8 * 64 * 4 = 24,576
    private static final int TOTAL_ACTION_SLOTS = TYPES * FLOORS * TILES * ROTATIONS;

    private final QTable qTable;
    private final Random random;
    
    // Learning parameters
    private double alpha = 0.1;
    private double gamma = 0.9;
    private double epsilon = 0.1;
    
    // Alignment constants
    private static final double GRID_ORIGIN_X = 200.0;
    private static final double GRID_ORIGIN_Y = 200.0;

    public QTableMultiDiscreteDecisionProvider() {
        this.qTable = new QTable(TOTAL_ACTION_SLOTS);
        this.random = new Random();
    }

    public QTableMultiDiscreteDecisionProvider(QTable qTable) {
        this.qTable = qTable;
        this.random = new Random();
    }

    @Override
    public MultiDiscretePhaseDecision provideDecision(MultiDiscretePhaseContext context) {
        long state = computeStateHash(context);
        
        List<Integer> candidates = generateCandidateIndices(context);
        
        // Strategy: choose among candidate phase combinations
        int chosenId = qTable.selectAction(state, candidates, epsilon);
        
        return decodeActionId(chosenId);
    }

    /**
     * Performs a Q-value update based on the reward received.
     */
    public void update(MultiDiscretePhaseContext current, MultiDiscreteAction action, double reward, MultiDiscretePhaseContext next) {
        long s = computeStateHash(current);
        long sNext = computeStateHash(next);
        
        int aIdx = encodeActionId(action.getTypeIndex(), action.getFloorIndex(), 
                                  action.getTileIndex(), action.getRotationIndex());
        
        qTable.updateQValue(s, aIdx, reward, sNext, alpha, gamma);
    }

    // --- Type-Aware Candidate Generation ---

    private List<Integer> generateCandidateIndices(MultiDiscretePhaseContext context) {
        List<Integer> candidates = new ArrayList<>();
        
        // 1. Get valid types for current state
        List<Integer> validTypes = ActionSpace.getValidTypeActions(context.isHasTC(), context.isHasLootRoom(), context.getStep());
        
        for (int typeIdx : validTypes) {
            // [STOP Semantics Fix]
            // Skip STOP for multi-discrete learning path until separate stop logic is designed.
            if (typeIdx == ActionSpace.STOP_TYPE_INDEX) {
                continue; 
            }

            // [HEURISTIC GUIDANCE]
            // Get sensible subsets based on component type to keep the tabular search space relevant
            int[] typeFloors = getCandidateFloorsForType(typeIdx);
            Set<Integer> typeTiles = getCandidateTilesForType(typeIdx, context.getGrid());
            int[] typeRotations = getCandidateRotationsForType(typeIdx);

            for (int tileIdx : typeTiles) {
                for (int floor : typeFloors) {
                    for (int rot : typeRotations) {
                        candidates.add(encodeActionId(typeIdx, floor, tileIdx, rot));
                    }
                }
            }
        }
        
        // Fallback: Safe central foundation placement instead of STOP
        if (candidates.isEmpty()) {
            candidates.add(encodeActionId(0, 0, 36, 0)); // Index 0 is Foundation, 36 is center
        }
        
        return candidates;
    }

    private int[] getCandidateFloorsForType(int typeIdx) {
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIdx);
        if (type == null) return new int[]{0};

        if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.FOUNDATION || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TRIANGLE_FOUNDATION) {
            return new int[]{0};
        }
        if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TC || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WORKBENCH || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.LOOT_ROOM) {
            return new int[]{0, 1, 2}; // Furniture restricted to lower floors
        }
        int[] allFloors = new int[MultiDiscreteActionSpace.FLOOR_COUNT];
        for (int i = 0; i < MultiDiscreteActionSpace.FLOOR_COUNT; i++) allFloors[i] = i;
        return allFloors; // Structural: all floors
    }

    private Set<Integer> getCandidateTilesForType(int typeIdx, com.rustbuilder.model.GridModel grid) {
        Set<Integer> occupied = new HashSet<>();
        List<BuildingBlock> blocks = grid.getAllBlocks();
        
        if (blocks.isEmpty()) {
            occupied.add(3 * 8 + 3);
            occupied.add(4 * 8 + 4);
            return occupied;
        }

        for (BuildingBlock b : blocks) {
            int tx = (int) Math.round((b.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int ty = (int) Math.round((b.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            if (tx >= 0 && tx < 8 && ty >= 0 && ty < 8) {
                occupied.add(tx * 8 + ty);
            }
        }

        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIdx);
        Set<Integer> neighbors = new HashSet<>();
        for (int tile : occupied) {
            int x = tile / 8;
            int y = tile % 8;
            if (x > 0) neighbors.add((x - 1) * 8 + y);
            if (x < 7) neighbors.add((x + 1) * 8 + y);
            if (y > 0) neighbors.add(x * 8 + (y - 1));
            if (y < 7) neighbors.add(x * 8 + (y + 1));
        }

        Set<Integer> result = new HashSet<>();
        if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.FOUNDATION || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TRIANGLE_FOUNDATION) {
            // Expansion: Neighbors that are NOT occupied
            for (int n : neighbors) {
                if (!occupied.contains(n)) result.add(n);
            }
            if (result.isEmpty()) result.addAll(neighbors);
        } else if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.TC || 
                   type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WORKBENCH || 
                   type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.LOOT_ROOM) {
            // Internal placement: Occupied tiles only
            result.addAll(occupied);
        } else {
            // Structural: standard structural adjacency
            result.addAll(neighbors);
        }

        if (result.isEmpty()) result.add(4 * 8 + 4); // Safety center
        return result;
    }

    private int[] getCandidateRotationsForType(int typeIdx) {
        com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType type = ActionSpace.decodeType(typeIdx);
        if (type == null) return new int[]{0};

        // For wall-like structures, use cardinal buckets to prune search space noise
        if (type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WALL || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.DOORWAY || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.WINDOW_FRAME || 
            type == com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.DOOR) {
            return new int[]{0, 1, 2, 3};
        }
        // Others: allow all rotations
        int[] allRotations = new int[MultiDiscreteActionSpace.ROTATION_COUNT];
        for (int i = 0; i < MultiDiscreteActionSpace.ROTATION_COUNT; i++) allRotations[i] = i;
        return allRotations;
    }

    // --- State & Action Mappings ---

    private long computeStateHash(MultiDiscretePhaseContext context) {
        long h = 0;
        List<BuildingBlock> blocks = new ArrayList<>(context.getGrid().getAllBlocks());
        
        // Stabilize hash by sorting blocks
        blocks.sort((b1, b2) -> {
            if (b1.getZ() != b2.getZ()) return Integer.compare(b1.getZ(), b2.getZ());
            int tx1 = (int) Math.round((b1.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int tx2 = (int) Math.round((b2.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            if (tx1 != tx2) return Integer.compare(tx1, tx2);
            int ty1 = (int) Math.round((b1.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            int ty2 = (int) Math.round((b2.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            if (ty1 != ty2) return Integer.compare(ty1, ty2);
            return b1.getType().name().compareTo(b2.getType().name());
        });

        for (BuildingBlock b : blocks) {
            int tx = (int) Math.round((b.getX() - GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
            int ty = (int) Math.round((b.getY() - GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
            int tz = b.getZ();
            if (tx >= 0 && tx < 8 && ty >= 0 && ty < 8) {
                // Combine type, coords into a unique value per block and mix it into the hash
                long blockBits = ((long)b.getType().ordinal() << 20) | (tx << 12) | (ty << 6) | tz;
                h = h * 31 + blockBits;
            }
        }
        h ^= (context.isHasTC() ? 0x5555555555555555L : 0);
        h ^= (context.isHasLootRoom() ? 0xAAAAAAAAAAAAAAAAL : 0);
        // Step removed from state hash to restore Markov property and enable generalization.
        return h;
    }

    private int encodeActionId(int type, int floor, int tileIdx, int rotIdx) {
        int id = type;
        id = id * FLOORS + floor;
        id = id * TILES + tileIdx;
        id = id * ROTATIONS + rotIdx;
        return id;
    }

    private MultiDiscretePhaseDecision decodeActionId(int id) {
        int rotIdx = id % ROTATIONS; id /= ROTATIONS;
        int tileIdx = id % TILES; id /= TILES;
        int floor = id % FLOORS; id /= FLOORS;
        int type = id % TYPES;
        
        // NOTE: aimSector is currently hardcoded to 12 (center) because it is not yet used in tabular RL.
        // The QTableMultiDiscreteDecisionProvider acts as a 5-phase provider inside the 5-phase contract.
        int fixedAimSector = 12;
        
        return new MultiDiscretePhaseDecision(type, floor, tileIdx, rotIdx, fixedAimSector);
    }

    public void setAlpha(double alpha) { this.alpha = alpha; }
    public void setGamma(double gamma) { this.gamma = gamma; }
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }
    public QTable getQTable() { return qTable; }
}
