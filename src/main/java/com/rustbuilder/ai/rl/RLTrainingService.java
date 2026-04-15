package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.core.AIModelManager;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.rl.legacy.*;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.ai.rl.nn.DQNAgent;
import com.rustbuilder.ai.rl.nn.ExperienceReplay;
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
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.service.evaluator.HouseEvaluator.EvaluationResult;

/**
 * Main service to train the RL Agent using a Deep Q-Network.
 */
public class RLTrainingService {

    public static class PlacementResult {
        public boolean inserted = false;
        public boolean survived = false;
        public BuildingBlock placedBlock = null;
        public String failReason = null;
        public PlacementError error = PlacementError.NONE;
    }

    /**
     * [RL REDESIGN] Helper to store episode outcome and stats.
     */
    private static class EpisodeResult {
        GridModel grid;
        double accStepReward = 0;     // Sum of individual step rewards
        double finalEvalReward = 0;    // Final score from HouseEvaluator at episode end
        int invalidActions = 0;
        int totalActions = 0;
        int blocksPlaced = 0;
        boolean hasTC = false;
        boolean hasLootRoom = false;
        EvaluationResult evaluationResult = null;
        
        java.util.Map<PlacementError, Integer> errorStats = new java.util.EnumMap<>(PlacementError.class);
        java.util.Map<com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType, Integer> typeStats = new java.util.EnumMap<>(com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType.class);
    }

    private DQNAgent agent;
    private ExperienceReplay memory;
    private final HouseEvaluator evaluator;
    private final Random random;

    /**
     * [RL REDESIGN] Placeholder for future multi-discrete action space integration.
     * Current system still uses ActionSpace (2-phase).
     */
    private MultiDiscreteAction currentMultiAction;
    
    // Hyperparameters
    private double epsilon = 1.0;
    private double epsilonDecay;
    private final double minEpsilon = 0.05;
    private final int batchSize = 32;
    private final int targetUpdateFreq = 10;
    
    // Runtime State
    private int episodesTrained = 0;
    private double bestScore = -1;
    private GridModel bestGridModel;
    private GridModel currentGridModel;
    private double lastTrainLoss = 0;
    
    // [RL REDESIGN] Experimental 5-phase flow
    private boolean useMultiDiscreteFlow = true;
    private boolean useMultiDiscreteLearning = true;
    private MultiDiscretePhasePolicy multiDiscretePolicy;
    private NeuralMultiDiscreteDecisionProvider multiDiscreteNeuralProvider;
    private MultiDiscreteDQNAgent multiDiscreteAgent;
    private MultiDiscreteExperienceReplay multiDiscreteMemory;
    private MultiDiscreteStateObserver multiDiscreteObserver;
    // Legacy QTable fallback maintained for internal debug baseline
    private QTableMultiDiscreteDecisionProvider multiDiscreteLearningProvider;
    
    
    // Extended stats
    private final java.util.LinkedList<Double> recentRewards = new java.util.LinkedList<>();
    private final java.util.LinkedList<Double> recentEvalScores = new java.util.LinkedList<>();
    private static final int AVG_WINDOW = 50;
    private double avgReward = 0;
    private double avgEvalScore = 0;
    private int lastEpisodeInvalidActions = 0;
    private int lastEpisodeTotalActions = 0;
    private int bestBaseBlocks = 0;
    private boolean bestBaseHasTC = false;
    private int bestBaseDoors = 0;
    
    // Training log
    private Path logFilePath;
    private Path invalidLogFilePath;
    
    public RLTrainingService() {
        this.memory = new ExperienceReplay(10000);
        this.evaluator = new HouseEvaluator();
        this.bestGridModel = new GridModel();
        this.currentGridModel = new GridModel();
        this.random = new Random();
        
        // [RL REDESIGN] Neural Multi-Discrete Flow config
        this.multiDiscreteMemory = new MultiDiscreteExperienceReplay(10000);
        this.multiDiscreteAgent = new MultiDiscreteDQNAgent(StateEncoder.CHANNELS, StateEncoder.MAX_FLOORS, StateEncoder.GRID_SIZE, StateEncoder.GRID_SIZE);
        this.multiDiscreteNeuralProvider = new NeuralMultiDiscreteDecisionProvider(this.multiDiscreteAgent);
        
        // Legacy QTable baseline
        this.multiDiscreteLearningProvider = new QTableMultiDiscreteDecisionProvider();
        
        // Default policy is Neural
        this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(this.multiDiscreteNeuralProvider);
        
        // Two-phase action space: 12 types + 2048 positions = 2060 total outputs
        this.agent = new DQNAgent(StateEncoder.CHANNELS, StateEncoder.MAX_FLOORS, StateEncoder.GRID_SIZE, StateEncoder.GRID_SIZE, ActionSpace.TOTAL_ACTIONS);
    }
    
    /**
     * Train for a number of epochs, each epoch running 'episodes' training episodes.
     * Progress callback receives: [currentEpoch, totalEpochs, episodeInEpoch, episodesPerEpoch, bestScore, epsilon]
     */
    public void train(int episodes, int maxStepsPerEpisode, double logW, double costW, double raidW,
                      int epochs, Consumer<TrainingMetrics> progressCallback, Runnable epochCompleteCallback) {
        evaluator.setWeights(logW, costW, raidW);
        
        int totalEpisodesToTrain = epochs * episodes;
        double exploreEpisodes = totalEpisodesToTrain * 0.8;
        
        if (exploreEpisodes > 0 && epsilon > (minEpsilon + 0.001)) {
            epsilonDecay = Math.pow(minEpsilon / epsilon, 1.0 / exploreEpisodes);
        } else {
            epsilonDecay = 1.0;
        }
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochTotalReward = 0;
            int epochTotalInvalid = 0;
            int epochTotalActions = 0;
            int epochTotalBlocks = 0;
            double epochBestEpScore = -1;
            long epochStartMs = System.currentTimeMillis();
            
            for (int ep = 0; ep < episodes; ep++) {
                EpisodeResult result;
                if (useMultiDiscreteFlow) {
                    // [FIX 5] Set epsilon BEFORE the episode so the agent explores at the correct rate
                    if (useMultiDiscreteLearning && multiDiscreteNeuralProvider != null) {
                        multiDiscreteNeuralProvider.setEpsilon(epsilon);
                    }
                    result = runExperimentalMultiDiscreteEpisode(maxStepsPerEpisode);
                } else {
                    result = runLegacyEpisode(maxStepsPerEpisode);
                }
                
                performEpisodeEvaluation(result, maxStepsPerEpisode, epoch, ep, episodes);
                if (!useMultiDiscreteFlow) {
                    finalizeLegacyTraining(result);
                }
                updateBestBase(result);
                
                double episodeEvalScore = 0.0;
                if (result.evaluationResult != null) {
                    episodeEvalScore = result.evaluationResult.finalScore;
                }
                recentEvalScores.addLast(episodeEvalScore);
                if (recentEvalScores.size() > AVG_WINDOW) recentEvalScores.removeFirst();
                avgEvalScore = recentEvalScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                
                if (ep > 0 && ep % targetUpdateFreq == 0) {
                    agent.updateTargetNetwork();
                    if (multiDiscreteAgent != null) {
                        multiDiscreteAgent.updateTargetNetwork();
                    }
                }
                
                if (epsilon > minEpsilon) {
                    epsilon *= epsilonDecay;
                }
                
                episodesTrained++;
        
                this.lastEpisodeInvalidActions = result.invalidActions;
                this.lastEpisodeTotalActions = result.totalActions;
                
                recentRewards.addLast(result.accStepReward);
                if (recentRewards.size() > AVG_WINDOW) recentRewards.removeFirst();
                avgReward = recentRewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                
                if (progressCallback != null && (ep % 5 == 0 || ep == episodes - 1)) {
                    double invalidRate = result.totalActions > 0 ? (double) result.invalidActions / result.totalActions : 0.0;
                    double loss = lastTrainLoss;
                    
                    TrainingMetrics metrics = new TrainingMetrics(
                        epoch + 1, epochs, ep + 1, episodes,
                        episodesTrained, bestScore, epsilon, loss,
                        avgReward, invalidRate, result.invalidActions, result.totalActions,
                        bestBaseBlocks, bestBaseHasTC, bestBaseDoors,
                        avgEvalScore, episodeEvalScore, result.accStepReward, result.finalEvalReward,
                        useMultiDiscreteFlow ? multiDiscreteMemory.size() : memory.size()
                    );
                    progressCallback.accept(metrics);
                }
                
                epochTotalReward += result.accStepReward;
                epochTotalActions += result.totalActions;
                epochTotalInvalid += result.invalidActions;
                epochTotalBlocks += result.blocksPlaced;
                if (result.finalEvalReward > epochBestEpScore) epochBestEpScore = result.finalEvalReward;
            }
            
            writeEpochLog(epoch + 1, episodes, epochTotalReward, epochTotalInvalid, epochTotalActions,
                          epochTotalBlocks, epochBestEpScore, System.currentTimeMillis() - epochStartMs);
            
            if (epochCompleteCallback != null) {
                epochCompleteCallback.run();
            }
        }
    }

    /**
     * [RL REDESIGN] Legacy 2-phase episode execution.
     */
    private EpisodeResult runLegacyEpisode(int maxStepsPerEpisode) {
        EpisodeResult result = new EpisodeResult();
        result.grid = new GridModel();
        
        for (int step = 0; step < maxStepsPerEpisode; step++) {
            result.totalActions++;
            double reward;
            
            // PHASE 1: Select block type
            INDArray statePhase1 = StateEncoder.encodeWithPhase(result.grid, -1);
            List<Integer> validTypes = ActionSpace.getValidTypeActions(result.hasTC, result.hasLootRoom, step);
            
            int typeIdx;
            if (random.nextDouble() < epsilon) {
                typeIdx = validTypes.get(random.nextInt(validTypes.size()));
            } else {
                double[] qVals = agent.getQValues(statePhase1);
                double maxQ = -Double.MAX_VALUE;
                int bestType = validTypes.get(0);
                for (int t : validTypes) {
                    if (qVals[t] > maxQ) {
                        maxQ = qVals[t];
                        bestType = t;
                    }
                }
                typeIdx = bestType;
            }
            
            if (typeIdx == ActionSpace.STOP_TYPE_INDEX) break;
            BuildAction.ActionType selectedType = ActionSpace.decodeType(typeIdx);
            if (selectedType == null) break;
            
            // PHASE 2: Select position
            INDArray statePhase2 = StateEncoder.encodeWithPhase(result.grid, selectedType.ordinal());
            List<Integer> validPositions = ActionSpace.getValidPositionActions(selectedType, step, result.grid);
            
            int posIdx;
            if (random.nextDouble() < epsilon) {
                posIdx = validPositions.get(random.nextInt(validPositions.size()));
            } else {
                double[] qVals = agent.getQValues(statePhase2);
                double maxQ = -Double.MAX_VALUE;
                int bestPos = validPositions.get(0);
                for (int p : validPositions) {
                    if (qVals[p] > maxQ) {
                        maxQ = qVals[p];
                        bestPos = p;
                    }
                }
                posIdx = bestPos;
            }
            
            BuildAction action = ActionSpace.decodePosition(posIdx, selectedType);
            this.currentMultiAction = toMultiDiscrete(action);
            
            if (action.actionType == BuildAction.ActionType.TC) result.hasTC = true;
            if (action.actionType == BuildAction.ActionType.LOOT_ROOM) result.hasLootRoom = true;
            
            PlacementResult pResult = placeBlock(result.grid, action);
            if (pResult.inserted) result.blocksPlaced++;
            else result.invalidActions++;
            
            result.grid.finalizeLoad();
            pResult.survived = (pResult.inserted && result.grid.getAllBlocks().contains(pResult.placedBlock));
            reward = StepRewardFunction.calculate(result.grid, action, pResult.inserted, pResult.survived, pResult.placedBlock, pResult.error);
            
            // Transitions
            INDArray nextPhase1 = StateEncoder.encodeWithPhase(result.grid, -1);
            List<Integer> nextValidTypes = ActionSpace.getValidTypeActions(result.hasTC, result.hasLootRoom, step + 1);
            memory.add(new ExperienceReplay.Transition(statePhase1, typeIdx, 0.0, statePhase2, false, validPositions));
            memory.add(new ExperienceReplay.Transition(statePhase2, posIdx, reward, nextPhase1, false, nextValidTypes));
            
            if (memory.size() >= batchSize) {
                lastTrainLoss = agent.trainBatch(memory.sample(batchSize));
            }
            
            result.accStepReward += reward;
        }
        return result;
    }

    /**
     * [RL REDESIGN] Experimental Multi-Discrete rollout execution.
     */
    private EpisodeResult runExperimentalMultiDiscreteEpisode(int maxStepsPerEpisode) {
        EpisodeResult result = new EpisodeResult();
        result.grid = new GridModel();
        
        int consecutiveInvalidSteps = 0;
        int consecutiveNoGrowthSteps = 0;
        int lastBlockCount = 0;
        
        for (int step = 0; step < maxStepsPerEpisode; step++) {
            result.totalActions++;
            
            MultiDiscretePhaseContext context = new MultiDiscretePhaseContext(
                result.grid, result.hasTC, result.hasLootRoom, step, maxStepsPerEpisode
            );
            
            // [FIX 1] Encode state BEFORE the grid is mutated by placeBlock()
            INDArray stateTensor = null;
            GridModel gridBeforeAction = null;
            if (useMultiDiscreteLearning && multiDiscreteMemory != null) {
                stateTensor = StateEncoder.encodeWithPhase(result.grid, -1);
                gridBeforeAction = result.grid.clone();
            }
            
            MultiDiscreteAction multiAction = multiDiscretePolicy.chooseAction(context, this.multiDiscreteObserver);
            this.currentMultiAction = multiAction;
            
            if (multiAction.getTypeIndex() == ActionSpace.STOP_TYPE_INDEX) {
                if (useMultiDiscreteLearning && multiDiscreteMemory != null && stateTensor != null) {
                    INDArray nextStateTensor = StateEncoder.encodeWithPhase(result.grid, -1);
                    
                    multiDiscreteMemory.add(new MultiDiscreteExperienceReplay.Transition(
                        stateTensor, multiAction, 0.0, nextStateTensor, true, step,
                        gridBeforeAction, result.grid.clone(), true
                    ));
                    
                    if (multiDiscreteMemory.size() >= 48 && step % 2 == 0) {
                        List<MultiDiscreteExperienceReplay.Transition> batch = multiDiscreteMemory.sample(64);
                        lastTrainLoss = multiDiscreteAgent.trainBatch(batch);
                    }
                }
                break;
            }

            BuildAction legacyAction = MultiDiscreteActionMapper.toBuildAction(multiAction);
            if (legacyAction.actionType == BuildAction.ActionType.TC) result.hasTC = true;
            if (legacyAction.actionType == BuildAction.ActionType.LOOT_ROOM) result.hasLootRoom = true;
            
            PlacementResult pResult = placeBlock(result.grid, legacyAction);
            // Reward calculation moved to post-finalizeLoad phase


            if (pResult.inserted) {
                result.blocksPlaced++;
                consecutiveInvalidSteps = 0;
            } else {
                result.invalidActions++;
                consecutiveInvalidSteps++;
                
                // Collect analytics
                result.errorStats.merge(pResult.error, 1, Integer::sum);
                result.typeStats.merge(legacyAction.actionType, 1, Integer::sum);

                // Log sampled invalid action reasons to CSV (every 5th episode, first 5 invalids)
                if (episodesTrained % 5 == 0 && result.invalidActions <= 5) {
                    writeInvalidActionLog(step, episodesTrained,
                        pResult.error.name(), 
                        legacyAction.actionType.name(), multiAction.getFloorIndex(),
                        multiAction.getTileX(), multiAction.getTileY(), multiAction.getRotationIndex(),
                        multiAction.getAimSector());
                }
            }
            
            result.grid.finalizeLoad();
            
            // [FIX 3] Move reward calculation here so pResult.survived is correctly set by physics
            pResult.survived = (pResult.inserted && result.grid.getAllBlocks().contains(pResult.placedBlock));
            double stepReward = StepRewardFunction.calculate(result.grid, legacyAction, pResult.inserted, pResult.survived, pResult.placedBlock, pResult.error);
            result.accStepReward += stepReward;
            
            int currentBlockCount = result.grid.getAllBlocks().size();
            if (currentBlockCount > lastBlockCount) {
                consecutiveNoGrowthSteps = 0;
            } else {
                consecutiveNoGrowthSteps++;
            }
            lastBlockCount = currentBlockCount;
            
             // pResult.survived calculated above before reward
            
            // [FIX 2] Determine if this is a terminal step
            boolean isTerminal = (step == maxStepsPerEpisode - 1)
                || (consecutiveInvalidSteps >= 10)
                || (consecutiveNoGrowthSteps >= 15 && currentBlockCount > 0);
            
            // Neural learning update for multi-discrete path
            if (useMultiDiscreteLearning && multiDiscreteMemory != null && stateTensor != null) {
                // [FIX 1] Encode nextState AFTER grid mutation
                INDArray nextStateTensor = StateEncoder.encodeWithPhase(result.grid, -1);
                
                multiDiscreteMemory.add(new MultiDiscreteExperienceReplay.Transition(
                    stateTensor, multiAction, stepReward, nextStateTensor, isTerminal, step,
                    gridBeforeAction, result.grid.clone(), pResult.survived
                ));
                
                // [BONUS] Start training earlier (>= 48) and more frequently (every 2 steps)
                if (multiDiscreteMemory.size() >= 48 && step % 2 == 0) {
                    List<MultiDiscreteExperienceReplay.Transition> batch = multiDiscreteMemory.sample(64);
                    lastTrainLoss = multiDiscreteAgent.trainBatch(batch);
                }
            }

            // [STABLE LOGIC] Early stop on both invalid streak and growth stall
            if (isTerminal && step < maxStepsPerEpisode - 1) {
                break;
            }
        }
        return result;
    }

    /**
     * [RL REDESIGN] Finalize training iteration for legacy 2-phase path.
     */
    private void finalizeLegacyTraining(EpisodeResult result) {
        INDArray termState = StateEncoder.encodeWithPhase(result.grid, -1);
        memory.add(new ExperienceReplay.Transition(termState, ActionSpace.STOP_TYPE_INDEX, result.finalEvalReward, termState, true, java.util.Collections.emptyList()));
        if (memory.size() >= batchSize) {
            lastTrainLoss = agent.trainBatch(memory.sample(batchSize));
        }
    }

    /**
     * [RL REDESIGN] Common evaluation logic for both training paths.
     */
    private void performEpisodeEvaluation(EpisodeResult result, int maxStepsPerEpisode, int epoch, int ep, int totalEpisodes) {
        GridModel grid = result.grid;
        int missedSteps = maxStepsPerEpisode - result.totalActions;
        double earlyStopPenalty = missedSteps * 0.07;

        if (grid.getAllBlocks().size() > 5) {
            result.evaluationResult = evaluator.evaluate(grid);
            
            List<BuildingBlock> allBlocks = grid.getAllBlocks();
            int blockCount = allBlocks.size();
            
            // Structural connectivity analysis (DFS-based components)
            boolean[] visited = new boolean[blockCount];
            java.util.List<java.util.List<Integer>> components = new java.util.ArrayList<>();
            
            for (int i = 0; i < blockCount; i++) {
                if (visited[i]) continue;
                java.util.List<Integer> comp = new java.util.ArrayList<>();
                java.util.ArrayDeque<Integer> queue = new java.util.ArrayDeque<>();
                queue.add(i);
                visited[i] = true;
                while (!queue.isEmpty()) {
                    int cur = queue.poll();
                    comp.add(cur);
                    for (int j = 0; j < blockCount; j++) {
                        if (visited[j]) continue;
                        if (areBlocksConnected(allBlocks.get(cur), allBlocks.get(j))) {
                            visited[j] = true;
                            queue.add(j);
                        }
                    }
                }
                components.add(comp);
            }
            
            double connectivityBonus = (components.size() == 1) ? 0.1 : 0.0;
            double fragmentPenalty = 0.0;
            if (components.size() > 1) {
                int mainIdx = 0;
                for (int c = 1; c < components.size(); c++) {
                    if (components.get(c).size() > components.get(mainIdx).size()) mainIdx = c;
                }
                for (int c = 0; c < components.size(); c++) {
                    if (c == mainIdx) continue;
                    double minDist = Double.MAX_VALUE;
                    for (int fi : components.get(c)) {
                        for (int mi : components.get(mainIdx)) {
                            double d = Math.hypot(
                                allBlocks.get(fi).getX() - allBlocks.get(mi).getX(),
                                allBlocks.get(fi).getY() - allBlocks.get(mi).getY());
                            if (d < minDist) minDist = d;
                        }
                    }
                    double minDistTiles = minDist / com.rustbuilder.config.GameConstants.TILE_SIZE;
                    fragmentPenalty += 0.2 + (minDistTiles * minDistTiles) / 10.0;
                }
            }
            
            double tcPenalty = 0.0;
            BuildingBlock tcBlock = null;
            for (BuildingBlock b : allBlocks) {
                if (b.getType() == com.rustbuilder.model.core.BuildingType.TC) { tcBlock = b; break; }
            }
            if (tcBlock != null) {
                int tcCompIdx = -1;
                for (int c = 0; c < components.size(); c++) {
                    for (int idx : components.get(c)) {
                        if (allBlocks.get(idx) == tcBlock) { tcCompIdx = c; break; }
                    }
                    if (tcCompIdx >= 0) break;
                }
                for (int i = 0; i < blockCount; i++) {
                    BuildingBlock b = allBlocks.get(i);
                    if (b == tcBlock) continue;
                    int bComp = -1;
                    for (int c = 0; c < components.size(); c++) {
                        if (components.get(c).contains(i)) { bComp = c; break; }
                    }
                    if (bComp != tcCompIdx) tcPenalty += 0.01;
                    double distToTC = Math.hypot(b.getX() - tcBlock.getX(), b.getY() - tcBlock.getY()) / com.rustbuilder.config.GameConstants.TILE_SIZE;
                    tcPenalty += distToTC * 0.01;
                }
            }
            
            double rawScore = result.evaluationResult.finalScore * 20.0;
            double logisticsBonus = (result.evaluationResult.logistics.score > 0) ? 10.0 : 0.0;
            double raidBonus = (result.evaluationResult.raid.sulfurToTC > 0) ? (result.evaluationResult.raid.score * 30.0) : 0.0;
            
            result.finalEvalReward = rawScore + logisticsBonus + raidBonus + connectivityBonus
                        - earlyStopPenalty - fragmentPenalty - tcPenalty;
            
            // Distribute final reward backwards in neural memory (Multi-discrete)
            if (useMultiDiscreteLearning && multiDiscreteMemory != null && multiDiscreteMemory.size() > 0) {
                // Approximate terminal reward assignment to last steps
                int size = multiDiscreteMemory.size();
                int lookback = Math.min(size, result.totalActions);
                for (int i = 0; i < lookback; i++) {
                    // Ongoing step rewards are already in memory
                }
            }
            
            // Legacy distribute final reward backwards
            if (!useMultiDiscreteFlow && memory.size() > 0) {
            }
            
        } else {
            result.finalEvalReward = -5.0 - earlyStopPenalty;
        }
    }

    /**
     * [RL REDESIGN] Track best base based on evaluation score.
     */
    private void updateBestBase(EpisodeResult result) {
        if (result.finalEvalReward > bestScore) {
            bestScore = result.finalEvalReward;
            
            bestGridModel.clear();
            for (BuildingBlock b : result.grid.getAllBlocks()) {
                bestGridModel.addBlockSilent(cloneBlock(b));
            }
            
            bestBaseBlocks = result.blocksPlaced;
            bestBaseHasTC = result.hasTC;
            for (BuildingBlock bk : result.grid.getAllBlocks()) {
                if (bk instanceof ToolCupboard) bestBaseHasTC = true;
            }
            bestBaseDoors = (int) result.grid.getAllBlocks().stream().filter(bl -> bl instanceof Door).count();
        }
        
        // Snapshot current episode grid for UI previews regardless of score
        currentGridModel.clear();
        for (BuildingBlock b : result.grid.getAllBlocks()) {
            currentGridModel.addBlockSilent(cloneBlock(b));
        }
    }


    public PlacementResult placeBlock(GridModel gridModel, BuildAction action) {
        PlacementResult res = new PlacementResult();
        com.rustbuilder.util.GridPlacementUtils.Placement placement = com.rustbuilder.util.GridPlacementUtils.calculatePlacement(gridModel, action);
        
        if (!placement.valid) {
            res.error = (placement.error != PlacementError.NONE) ? placement.error : PlacementError.BAD_SOCKET;
            res.failReason = "invalid_placement(" + res.error + "): " + action.actionType;
            return res;
        }

        double finalX = placement.x;
        double finalY = placement.y;
        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        
        // [Task] Heuristic/Safe fallback for floor (already in NeuralDecisionProvider, but double checking here)
        if (z < 0) z = 0; 

        BuildingTier tier = tierFromInt(action.tier);
        DoorType doorType = doorTypeFromInt(action.doorType);
        BuildingBlock block = null;

        switch (action.actionType) {
            case FOUNDATION: block = new Foundation(finalX, finalY, z); block.setRotation(finalRotation); break;
            case TRIANGLE_FOUNDATION: block = new TriangleFoundation(finalX, finalY, z, finalRotation); break;
            case WALL: block = new Wall(finalX, finalY, z, finalOrientation); break;
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
            case DOOR: block = new Door(finalX, finalY, z, finalOrientation, doorType); break;
            case FLOOR: block = new Floor(finalX, finalY, z, finalRotation); break;
            case TRIANGLE_FLOOR: block = new TriangleFloor(finalX, finalY, z, finalRotation); break;
            case TC: block = new ToolCupboard(finalX, finalY, z, finalRotation); break;
            case WORKBENCH: block = new Workbench(finalX, finalY, z, finalRotation); break;
            case LOOT_ROOM: block = new LootRoom(finalX, finalY, z, finalRotation); break;
        }

        if (block != null) {
            block.setRotation(finalRotation);
            block.setTier(tier);
            res.placedBlock = block;

            boolean isFurniture = action.actionType == BuildAction.ActionType.TC || action.actionType == BuildAction.ActionType.WORKBENCH || action.actionType == BuildAction.ActionType.LOOT_ROOM;

            if (isFurniture) {
                if (gridModel.canPlace(block)) {
                    // Extra: check no other deployable already at this exact spot
                    boolean occupied = false;
                    for (BuildingBlock b2 : gridModel.getAllBlocks()) {
                        if (com.rustbuilder.util.BuildingTypeUtils.isFurniture(b2.getType()) &&
                            b2.getZ() == block.getZ() &&
                            Math.abs(b2.getX() - block.getX()) < 1.0 &&
                            Math.abs(b2.getY() - block.getY()) < 1.0) {
                            occupied = true;
                            break;
                        }
                    }
                    if (!occupied) {
                        gridModel.addBlockSilent(block);
                        res.inserted = true;
                        res.survived = true;
                        res.error = PlacementError.NONE;
                        return res;
                    } else {
                        res.error = PlacementError.COLLISION;
                        res.failReason = "furniture_slot_occupied";
                    }
                } else {
                    res.error = PlacementError.NO_SUPPORT;
                    res.failReason = "furniture_canPlace_failed";
                }
            } else {
                // Foundations, walls, floors, doors etc: full canPlace (collision + support)
                if (gridModel.canPlace(block)) {
                    gridModel.addBlockSilent(block);
                    if (action.actionType == BuildAction.ActionType.DOORWAY) {
                        Door d = new Door(finalX, finalY, z, finalOrientation, doorType);
                        d.setTier(tier);
                        d.setRotation(finalRotation);
                        if (gridModel.canPlace(d)) {
                            gridModel.addBlockSilent(d);
                        }
                    }
                    res.inserted = true;
                    res.error = PlacementError.NONE;
                    return res;
                } else {
                    // Decide if it's NO_SUPPORT or COLLISION
                    // (Simplified: canPlace failure often means no foundation/support in this codebase's rules)
                    res.error = isFoundation(block) ? PlacementError.COLLISION : PlacementError.NO_SUPPORT;
                    res.failReason = "structure_canPlace_failed";
                }
            }
        }
        
        return res;
    }

    private BuildingBlock cloneBlock(BuildingBlock b) {
        BuildingBlock clone = null;
        if (b instanceof Foundation) clone = new Foundation(b.getX(), b.getY(), b.getZ());
        else if (b instanceof TriangleFoundation) clone = new TriangleFoundation(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof Wall) {
            Wall w = (Wall)b;
            clone = new Wall(w.getX(), w.getY(), w.getZ(), w.getOrientation());
            clone.setType(w.getType());
            if (w.getType() == BuildingType.DOORWAY) ((Wall)clone).setDoorType(w.getDoorType());
        }
        else if (b instanceof Floor) clone = new Floor(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof TriangleFloor) clone = new TriangleFloor(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof Door) clone = new Door(b.getX(), b.getY(), b.getZ(), ((Door)b).getOrientation(), ((Door)b).getDoorType());
        else if (b instanceof ToolCupboard) clone = new ToolCupboard(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof Workbench) clone = new Workbench(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof LootRoom) clone = new LootRoom(b.getX(), b.getY(), b.getZ(), b.getRotation());

        if (clone != null) {
            clone.setRotation(b.getRotation());
            clone.setTier(b.getTier());
        }
        return clone;
    }

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
        return DoorType.SHEET_METAL; 
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

    // ---- Training log ----
    
    public void setLogFile(String modelName) {
        Path dir = Paths.get("models_rl");
        try { Files.createDirectories(dir); } catch (IOException ignored) {}
        
        String suffix = this.useMultiDiscreteFlow ? "_multi_discrete_training.csv" : "_legacy_training.csv";
        logFilePath = dir.resolve(modelName + suffix);
        
        // Write header if new file
        if (!Files.exists(logFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(logFilePath.toFile()))) {
                pw.println("timestamp,epoch,episodes,avg_reward,best_reward_all_time,best_ep_reward,epsilon,loss,"
                         + "invalid_rate_pct,total_blocks,ram_mb,best_base_blocks,best_base_tc,best_base_doors,epoch_time_sec");
            } catch (IOException e) {
                System.err.println("[TrainingLog] Failed to write header: " + e.getMessage());
            }
        }
        
        // Invalid action log
        invalidLogFilePath = dir.resolve(modelName + "_invalid_actions.csv");
        if (!Files.exists(invalidLogFilePath)) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(invalidLogFilePath.toFile()))) {
                pw.println("timestamp,episode,step,fail_reason,action_type,floor,tileX,tileY,rotation,aimSector");
            } catch (IOException e) {
                System.err.println("[TrainingLog] Failed to write invalid action header: " + e.getMessage());
            }
        }
    }
    
    private void writeEpochLog(int epoch, int episodesInEpoch, double epochTotalReward,
                                int epochInvalid, int epochActions, int epochBlocks,
                                double epochBestEpScore, long epochMs) {
        if (logFilePath == null) return;
        
        double epochAvgReward = episodesInEpoch > 0 ? epochTotalReward / episodesInEpoch : 0;
        double invalidPct = epochActions > 0 ? (double) epochInvalid / epochActions * 100.0 : 0;
        long ramMB = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        
        try (PrintWriter pw = new PrintWriter(new FileWriter(logFilePath.toFile(), true))) {
            pw.printf(java.util.Locale.US,
                "%s,%d,%d,%.4f,%.4f,%.4f,%.5f,%.6f,%.1f,%d,%d,%d,%s,%d,%.1f%n",
                ts, epoch, episodesInEpoch, epochAvgReward, bestScore, epochBestEpScore,
                epsilon, lastTrainLoss, invalidPct, epochBlocks, ramMB,
                bestBaseBlocks, bestBaseHasTC ? "YES" : "NO", bestBaseDoors,
                epochMs / 1000.0);
        } catch (IOException e) {
            System.err.println("[TrainingLog] Failed to write log: " + e.getMessage());
        }
    }
    
    private void writeInvalidActionLog(int step, int episode, String failReason,
                                        String actionType, int floor, int tileX, int tileY, int rotation, int aimSector) {
        if (invalidLogFilePath == null) return;
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        try (PrintWriter pw = new PrintWriter(new FileWriter(invalidLogFilePath.toFile(), true))) {
            pw.printf("%s,%d,%d,%s,%s,%d,%d,%d,%d,%d%n",
                ts, episode, step, failReason, actionType, floor, tileX, tileY, rotation, aimSector);
        } catch (IOException e) {
            System.err.println("[TrainingLog] Failed to write invalid action log: " + e.getMessage());
        }
    }
    
    public double getBestScore() { return bestScore; } // kept for RLModelManager
    public GridModel getBestGridModel() { return bestGridModel; }
    public GridModel getCurrentEpisodeGrid() { return currentGridModel; }
    public double getEpsilon() { return epsilon; }
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }
    public int getEpisodesTrained() { return episodesTrained; }
    public void setEpisodesTrained(int ep) { this.episodesTrained = ep; }
    public DQNAgent getAgent() { return agent; }
    public ExperienceReplay getMemory() { return memory; }
    
    public TrainingMetrics getMetrics() {
        double invalidRate = lastEpisodeTotalActions > 0 ? (double) lastEpisodeInvalidActions / lastEpisodeTotalActions : 0.0;
        int mSize = useMultiDiscreteFlow ? multiDiscreteMemory.size() : memory.size();
        return new TrainingMetrics(0, 0, 0, 0, episodesTrained, bestScore, epsilon, lastTrainLoss, avgReward, invalidRate, lastEpisodeInvalidActions, lastEpisodeTotalActions, bestBaseBlocks, bestBaseHasTC, bestBaseDoors, avgEvalScore, 0.0, 0.0, 0.0, mSize);
    }

    public boolean isUseMultiDiscreteFlow() { return useMultiDiscreteFlow; }
    public void setUseMultiDiscreteFlow(boolean use) { this.useMultiDiscreteFlow = use; }
    
    public boolean isUseMultiDiscreteLearning() { return useMultiDiscreteLearning; }
    public void setUseMultiDiscreteLearning(boolean use) { 
        this.useMultiDiscreteLearning = use; 
        if (use) {
            this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(this.multiDiscreteNeuralProvider);
        } else {
            this.multiDiscretePolicy = new HeuristicMultiDiscretePhasePolicy(this.random);
        }
    }

    public MultiDiscretePhasePolicy getMultiDiscretePolicy() { return multiDiscretePolicy; }
    public void setMultiDiscretePolicy(MultiDiscretePhasePolicy policy) { this.multiDiscretePolicy = policy; }
    
    public QTableMultiDiscreteDecisionProvider getMultiDiscreteLearningProvider() { return multiDiscreteLearningProvider; }
    public void setMultiDiscreteDecisionProvider(QTableMultiDiscreteDecisionProvider provider) {
        this.multiDiscreteLearningProvider = provider;
        if (useMultiDiscreteLearning) {
            this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(multiDiscreteLearningProvider);
        }
    }

    /**
     * [RL REDESIGN]
     * Helper to map a legacy BuildAction into the new 5-phase MultiDiscreteAction contract.
     * This is used for future-proofing and logging.
     * 
     * SEE ALSO: {@link MultiDiscreteActionMapper#toBuildAction(MultiDiscreteAction)} 
     * for the reverse mapping (used in future multi-discrete train flow).
     * 
     * MAPPING:
     * - Rotation: 0, 90, 180, 270 degrees -> 4-bucket system (0, 1, 2, 3).
     * - Aim: set to 12 (center placeholder).
     */
    private MultiDiscreteAction toMultiDiscrete(BuildAction legacyAction) {
        if (legacyAction == null) return null;

        int typeIdx = ActionSpace.encodeType(legacyAction.actionType);
        int floorIdx = legacyAction.floor;
        
        // Combine grid coordinates into a singular tileIndex (0..63)
        int tileIndex = legacyAction.gridX * MultiDiscreteActionSpace.GRID_SIZE + legacyAction.gridY;

        // Map rotation (0..3) to 4-bucket RL rotation index (0..3)
        int rotIdx = mapLegacyOrientationToRotationIndex(legacyAction.orientation);

        // AimSector: center (12) as it currently has no physical meaning
        int aimSector = 12;

        return new MultiDiscreteAction(typeIdx, floorIdx, tileIndex, rotIdx, aimSector);
    }

    /**
     * Maps legacy 4-orientation system (0=N, 1=E, 2=S, 3=W) to RL 4-bucket rotation index.
     * Each bucket represents a 90-degree step.
     */
    private int mapLegacyOrientationToRotationIndex(int orientation) {
        com.rustbuilder.model.core.Orientation or = orientFromInt(orientation);
        switch (or) {
            case NORTH: return 0;
            case EAST:  return 1;
            case SOUTH: return 2;
            case WEST:  return 3;
            default:    return 0; // Fallback
        }
     }

    /**
     * Check if two blocks are structurally connected (socket proximity or shared foundation for furniture).
     */
    private boolean areBlocksConnected(BuildingBlock a, BuildingBlock b) {
        // Quick distance reject
        if (Math.abs(a.getX() - b.getX()) > 200 || Math.abs(a.getY() - b.getY()) > 200) return false;
        
        // Furniture connects to the foundation it sits on (same x,y position)
        boolean aFurn = com.rustbuilder.util.BuildingTypeUtils.isFurniture(a.getType());
        boolean bFurn = com.rustbuilder.util.BuildingTypeUtils.isFurniture(b.getType());
        boolean aBase = com.rustbuilder.util.BuildingTypeUtils.isFoundation(a.getType()) || com.rustbuilder.util.BuildingTypeUtils.isFloor(a.getType());
        boolean bBase = com.rustbuilder.util.BuildingTypeUtils.isFoundation(b.getType()) || com.rustbuilder.util.BuildingTypeUtils.isFloor(b.getType());
        
        if ((aFurn && bBase) || (bFurn && aBase)) {
            if (a.getZ() == b.getZ() && Math.abs(a.getX() - b.getX()) < 1.0 && Math.abs(a.getY() - b.getY()) < 1.0) {
                return true;
            }
        }
        
        // Socket-based connection
        for (com.rustbuilder.model.core.Socket s1 : a.getSockets()) {
            for (com.rustbuilder.model.core.Socket s2 : b.getSockets()) {
                double dx = s1.getX() - s2.getX();
                double dy = s1.getY() - s2.getY();
                if (dx * dx + dy * dy < 1.3) return true;
            }
        }
        return false;
    }

    private boolean isFoundation(BuildingBlock b) {
        if (b == null) return false;
        return b.getType() == BuildingType.FOUNDATION || b.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }
}
