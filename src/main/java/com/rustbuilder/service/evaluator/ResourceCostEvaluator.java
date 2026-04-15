package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;
import com.rustbuilder.service.*;
import com.rustbuilder.service.graph.*;
import com.rustbuilder.service.physics.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.ResourceType;

/**
 * Evaluates total resource cost of all placed blocks.
 * Score is normalized: lower cost = better score.
 */
public class ResourceCostEvaluator {

    // Reference cost for normalization (typical small base)
    // Increased to 35000 so the AI isn't overly punished for placing walls
    private static final double BASE_COST = 35000.0;

    public static class CostResult {
        public final Map<ResourceType, Integer> totalCost;
        public final double score; // 0.0 - 1.0, higher = cheaper

        public CostResult(Map<ResourceType, Integer> totalCost, double score) {
            this.totalCost = totalCost;
            this.score = score;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("Resources: ");
            totalCost.forEach((k, v) -> sb.append(k).append("=").append(v).append(" "));
            sb.append("| Score=").append(String.format("%.2f", score));
            return sb.toString();
        }
    }

    public CostResult evaluate(List<BuildingBlock> blocks) {
        Map<ResourceType, Integer> totalCost = new HashMap<>();

        for (BuildingBlock block : blocks) {
            Map<ResourceType, Integer> blockCost = block.getBuildCost();
            blockCost.forEach((res, amount) -> totalCost.merge(res, amount, Integer::sum));
        }

        // Convert to a single comparable number (weighted sum)
        double weightedTotal = 0;
        for (Map.Entry<ResourceType, Integer> entry : totalCost.entrySet()) {
            double weight = getResourceWeight(entry.getKey());
            weightedTotal += entry.getValue() * weight;
        }

        // Score: 1 / (1 + total/BASE_COST). Lower cost → higher score.
        double score = 1.0 / (1.0 + weightedTotal / BASE_COST);

        return new CostResult(totalCost, score);
    }

    /**
     * Relative "value" weight for each resource type.
     * Higher tier resources are more expensive to gather.
     */
    private double getResourceWeight(ResourceType type) {
        switch (type) {
            case WOOD:  return 1.0;
            case STONE: return 1.0;
            case METAL: return 2.5; 
            case HQM:   return 10.0;
            default:    return 1.0;
        }
    }
}
