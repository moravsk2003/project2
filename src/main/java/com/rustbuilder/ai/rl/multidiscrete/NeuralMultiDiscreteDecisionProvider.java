package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.rl.legacy.StateEncoder;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Provides multi-discrete actions using a trained neural network with conditional head architecture.
 * Uses real-time conditional masking to ensure high-quality valid actions during both exploration and exploitation.
 */
public class NeuralMultiDiscreteDecisionProvider implements MultiDiscretePhaseDecisionProvider {

    private final MultiDiscreteDQNAgent agent;
    private final Random random;
    private double epsilon = 0.1;

    public NeuralMultiDiscreteDecisionProvider(MultiDiscreteDQNAgent agent) {
        this.agent = Objects.requireNonNull(agent, "Agent cannot be null");
        this.random = new Random();
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public MultiDiscretePhaseDecision provideDecision(MultiDiscretePhaseContext context) {
        INDArray state = StateEncoder.encodeWithPhase(context.getGrid(), -1);
        int[] selected = new int[5];
        
        INDArray stateFeatures = agent.encodeStateFeatures(state);

        // Roll explore vs exploit once for the entire action
        boolean explore = random.nextDouble() < epsilon;

        // 1. PHASE: TYPE (Global step constraints)
        List<Integer> validTypes = HeuristicMaskingUtils.getFeasibleTypes(context.getGrid(), context.isHasTC(), context.isHasLootRoom(), context.getStep());
        
        if (validTypes.isEmpty()) {
            System.err.println("[RL Pipeline] Feasible types empty! Forcing STOP action.");
            if (state != null && !state.wasClosed()) state.close();
            return new MultiDiscretePhaseDecision(com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX, 0, 0, 0, 12);
        }

        if (explore) {
            selected[0] = validTypes.get(random.nextInt(validTypes.size()));
        } else {
            INDArray typeLogits = agent.predictType(stateFeatures);
            selected[0] = maskedArgmax(typeLogits, 0, validTypes);
        }

        if (selected[0] == com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX) {
            System.out.println("[COND-HEAD DEBUG] Terminal STOP picked.");
            if (state != null && !state.wasClosed()) state.close();
            return new MultiDiscretePhaseDecision(selected[0], 0, 0, 0, 12);
        }

        // 2. PHASE: FLOOR (Conditional on Type)
        List<Integer> validFloors = HeuristicMaskingUtils.getValidFloors(context.getGrid(), selected[0], context.getStep());

        if (explore) {
            selected[1] = validFloors.get(random.nextInt(validFloors.size()));
        } else {
            INDArray typeContext = ActionConditioningUtils.oneHot(selected[0], MultiDiscreteActionSpace.TYPE_COUNT);
            INDArray logits = agent.predictFloor(ActionConditioningUtils.concat(stateFeatures, typeContext));
            selected[1] = maskedArgmax(logits, 0, validFloors);
        }

        // 3. PHASE: TILE (Conditional on Type, Floor)
        List<Integer> validTiles = HeuristicMaskingUtils.getValidTiles(context.getGrid(), selected[0], selected[1], context.getStep());

        if (explore) {
            selected[2] = validTiles.get(random.nextInt(validTiles.size()));
        } else {
            INDArray typeContext = ActionConditioningUtils.oneHot(selected[0], MultiDiscreteActionSpace.TYPE_COUNT);
            INDArray floorContext = ActionConditioningUtils.oneHot(selected[1], MultiDiscreteActionSpace.FLOOR_COUNT);
            INDArray logits = agent.predictTile(ActionConditioningUtils.concat(stateFeatures, typeContext, floorContext));
            selected[2] = maskedArgmax(logits, 0, validTiles);
        }

        // 4. PHASE: ROTATION (Conditional on Type, Floor, Tile)
        List<Integer> validRotations = HeuristicMaskingUtils.getValidRotations(context.getGrid(), selected[0], selected[1], selected[2]);

        if (explore) {
            selected[3] = validRotations.get(random.nextInt(validRotations.size()));
        } else {
            INDArray typeContext = ActionConditioningUtils.oneHot(selected[0], MultiDiscreteActionSpace.TYPE_COUNT);
            INDArray floorContext = ActionConditioningUtils.oneHot(selected[1], MultiDiscreteActionSpace.FLOOR_COUNT);
            INDArray tileContext = ActionConditioningUtils.oneHot(selected[2], MultiDiscreteActionSpace.TILE_COUNT);
            INDArray logits = agent.predictRot(ActionConditioningUtils.concat(stateFeatures, typeContext, floorContext, tileContext));
            selected[3] = maskedArgmax(logits, 0, validRotations);
        }

        // 5. PHASE: AIM (Frozen)
        selected[4] = 12;

        System.out.println(String.format("[COND-HEAD DEBUG] Phase choices: Type=%d, Floor=%d, Tile=%d, Rot=%d, Aim=%d | MaskPrecision: %d valid tiles",
            selected[0], selected[1], selected[2], selected[3], selected[4], validTiles.size()));

        if (state != null && !state.wasClosed()) state.close();
        return new MultiDiscretePhaseDecision(selected[0], selected[1], selected[2], selected[3], selected[4]);
    }

    private int maskedArgmax(INDArray headOutput, int row, List<Integer> validIndices) {
        double maxQ = -Double.MAX_VALUE;
        int best = validIndices.get(0);
        for (int idx : validIndices) {
            double q = headOutput.getDouble(row, idx);
            if (q > maxQ) {
                maxQ = q;
                best = idx;
            }
        }
        return best;
    }
}
