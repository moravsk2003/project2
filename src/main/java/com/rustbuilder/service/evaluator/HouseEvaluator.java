package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;
import com.rustbuilder.service.*;
import com.rustbuilder.service.graph.*;
import com.rustbuilder.service.physics.*;

import java.util.List;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.GridModel;

/**
 * Combines all 3 evaluation criteria into a single score.
 * This will serve as the fitness function for the AI generator.
 */
public class HouseEvaluator {

    private double logisticsWeight = 0.3;
    private double costWeight = 0.3;
    private double raidWeight = 0.4;

    private final LogisticsEvaluator logisticsEvaluator = new LogisticsEvaluator();
    private final ResourceCostEvaluator costEvaluator = new ResourceCostEvaluator();
    private final RaidResistanceEvaluator raidEvaluator = new RaidResistanceEvaluator();

    public static class EvaluationResult {
        public final LogisticsEvaluator.LogisticsResult logistics;
        public final ResourceCostEvaluator.CostResult cost;
        public final RaidResistanceEvaluator.RaidResult raid;
        public final double finalScore;

        public EvaluationResult(LogisticsEvaluator.LogisticsResult logistics,
                                ResourceCostEvaluator.CostResult cost,
                                RaidResistanceEvaluator.RaidResult raid,
                                double finalScore) {
            this.logistics = logistics;
            this.cost = cost;
            this.raid = raid;
            this.finalScore = finalScore;
        }

        @Override
        public String toString() {
            return String.format(
                "=== House Evaluation ===\n%s\n%s\n%s\nFinal Score: %.2f",
                logistics, cost, raid, finalScore
            );
        }
    }

    public void setWeights(double logistics, double cost, double raid) {
        this.logisticsWeight = logistics;
        this.costWeight = cost;
        this.raidWeight = raid;
    }

    /**
     * Evaluate the house described by the GridModel.
     */
    public EvaluationResult evaluate(GridModel gridModel) {
        List<BuildingBlock> blocks = gridModel.getAllBlocks();

        // 1. Build 3D graph
        HouseGraph graph = new HouseGraph();
        graph.buildGraph(blocks);

        // 2. Evaluate all 3 criteria
        LogisticsEvaluator.LogisticsResult logResult = logisticsEvaluator.evaluate(graph);
        ResourceCostEvaluator.CostResult costResult = costEvaluator.evaluate(blocks);
        
        // Always calculate raid score so the reward surface isn't flat when TC is missing
        RaidResistanceEvaluator.RaidResult computedRaid = raidEvaluator.evaluate(graph, blocks);
        RaidResistanceEvaluator.RaidResult raidResult;
        
        if (logResult.score < 0.0001) {
             // heavily penalize but retain gradient
             raidResult = new RaidResistanceEvaluator.RaidResult(computedRaid.sulfurToTC, computedRaid.sulfurToLootRoom, computedRaid.score * 0.1);
        } else {
             raidResult = computedRaid;
        }

        // 3. Combine scores
        // logistics: higher = better (shorter paths)
        // cost: higher = cheaper (less resources)
        // raid: higher = more raid-resistant
        double finalScore = logisticsWeight * logResult.score +
                            costWeight * costResult.score +
                            raidWeight * raidResult.score;

        return new EvaluationResult(logResult, costResult, raidResult, finalScore);
    }
}
