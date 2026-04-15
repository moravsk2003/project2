package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;
import com.rustbuilder.service.*;
import com.rustbuilder.service.graph.*;
import com.rustbuilder.service.physics.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.graph.HouseGraph.TileEdge;
import com.rustbuilder.service.graph.HouseGraph.TileNode;
import com.rustbuilder.service.evaluator.RaidConstants;

/**
 * Evaluates raid resistance: minimum sulfur to reach TC from outside.
 * Uses Dijkstra on the raid-cost graph, considering splash damage.
 *
 * Splash damage: when multiple adjacent walls share a corner point,
 * a rocket can damage all of them simultaneously, reducing per-wall cost.
 */
public class RaidResistanceEvaluator {

    // Maximum realistic sulfur cost for a well-designed small/medium base
    // Set to 30,000 (enough for honeycomb + full coverage)
    private static final double MAX_EXPECTED_SULFUR = 30000.0;

    public static class RaidResult {
        public final int sulfurToTC;
        public final int sulfurToLootRoom;
        public final double score; // 0.0 - 1.0, higher = harder to raid

        public RaidResult(int sulfurToTC, int sulfurToLootRoom, double score) {
            this.sulfurToTC = sulfurToTC;
            this.sulfurToLootRoom = sulfurToLootRoom;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format("Raid: TC=%d sulfur, LootRoom=%d sulfur | Score=%.2f",
                    sulfurToTC, sulfurToLootRoom, score);
        }
    }

    public RaidResult evaluate(HouseGraph graph, List<BuildingBlock> allBlocks) {
        TileNode outside = graph.getOutsideNode();
        TileNode tc = graph.findNodeByType("tc");
        List<TileNode> lootRooms = graph.findAllNodesByType("loot_room");

        // Pre-compute splash groups for cheaper costs
        Map<BuildingBlock, Integer> splashCosts = computeSplashCosts(graph, allBlocks);

        // Dijkstra from outside using sulfur costs
        Map<NodeKey, Integer> sulfurDist = dijkstraRaid(graph, outside, splashCosts);

        int sulfurToTC = tc != null ? sulfurDist.getOrDefault(tc.id, Integer.MAX_VALUE) : Integer.MAX_VALUE;

        // Find minimum sulfur across ALL loot rooms (weakest link)
        int minSulfurToLR = Integer.MAX_VALUE;
        for (TileNode lr : lootRooms) {
            int s = sulfurDist.getOrDefault(lr.id, Integer.MAX_VALUE);
            if (s < minSulfurToLR) {
                minSulfurToLR = s;
            }
        }

        // --- NEW: Coverage Heuristic ---
        // Reward partial protection (walls around TC/Loot) even if the room isn't fully sealed.
        // This helps the genetic algorithm "climb the hill" from 0 protection to sealed protection.
        int tcCoverage = 0;
        if (tc != null) {
            TileNode tcTile = getUnderlyingTile(graph, tc);
            tcCoverage = calculateLocalCoverage(graph, tcTile, splashCosts);
        }
        
        int lrCoverage = 0;
        for (TileNode lr : lootRooms) {
            TileNode lrTile = getUnderlyingTile(graph, lr);
            lrCoverage += calculateLocalCoverage(graph, lrTile, splashCosts);
        }

        // Combined score formula:
        // combinedSulfur = min(TC, LR) + TC/10 + max(TC, LR)/10 + coverage heuristic
        double score;
        boolean hasTC = sulfurToTC < Integer.MAX_VALUE;
        boolean hasLR = minSulfurToLR < Integer.MAX_VALUE;

        if (!hasTC && !hasLR) {
            score = 0.0; // No key targets — invalid base
        } else {
            int effTC = hasTC ? sulfurToTC : 0;
            int effLR = hasLR ? minSulfurToLR : 0;

            double combinedSulfur;
            if (hasTC && hasLR) {
                combinedSulfur = Math.min(effTC, effLR) 
                    + effTC / 10.0 
                    + Math.max(effTC, effLR) / 10.0;
            } else if (hasTC) {
                combinedSulfur = effTC + effTC / 10.0; // only TC, no LR bonus
            } else {
                combinedSulfur = effLR; // only LR, no TC bonus
            }

            // Add coverage bonus (weighted lower so real raid cost is still dominant, but noticeable)
            // 0.25 factor: 1 stone wall (4400) -> +1100 score. 4 walls -> +4400 score.
            // Sealing the base provides the path cost (4400), creating a massive spike on the 4th wall.
            combinedSulfur += (tcCoverage + lrCoverage) * 0.25;

            score = Math.min(1.0, combinedSulfur / MAX_EXPECTED_SULFUR);
        }

        return new RaidResult(
                sulfurToTC == Integer.MAX_VALUE ? -1 : sulfurToTC,
                minSulfurToLR == Integer.MAX_VALUE ? -1 : minSulfurToLR,
                score
        );
    }

    private TileNode getUnderlyingTile(HouseGraph graph, TileNode furnitureNode) {
        if (furnitureNode == null) return null;
        for (TileEdge edge : graph.getEdges(furnitureNode)) {
            if ("tile".equals(edge.to.type)) {
                return edge.to;
            }
        }
        return null;
    }

    /**
     * Calculates the sum of sulfur costs of all structures immediately protecting the node.
     * Used as a heuristic to reward partial protection.
     */
    private int calculateLocalCoverage(HouseGraph graph, TileNode node, Map<BuildingBlock, Integer> splashCosts) {
        if (node == null) return 0;
        int coverage = 0;
        
        // Sum cost of all blocking structures on edges connected to this node
        for (TileEdge edge : graph.getEdges(node)) {
            if (edge.blocker != null) {
                // Use splash cost if available (more realistic), otherwise raw cost
                int cost = splashCosts.getOrDefault(edge.blocker, edge.raidSulfurCost);
                coverage += cost;
            }
        }
        return coverage;
    }

    /**
     * Compute splash-adjusted sulfur costs for walls.
     * Walls that share a corner with other walls can be destroyed cheaper with rockets.
     */
    private Map<BuildingBlock, Integer> computeSplashCosts(HouseGraph graph, List<BuildingBlock> allBlocks) {
        Map<BuildingBlock, Integer> costs = new HashMap<>();

        // Group walls by corner position
        // Two walls "share a corner" if they are on the same or adjacent tiles at the same Z
        Map<String, List<BuildingBlock>> cornerGroups = new HashMap<>();

        for (BuildingBlock block : allBlocks) {
            if (isWallType(block)) {
                // Use tile position + z as grouping key
                String key = String.format("%.0f_%.0f_%d", block.getX(), block.getY(), block.getZ());
                cornerGroups.computeIfAbsent(key, k -> new ArrayList<>()).add(block);
            }
        }

        // For each group, compute splash cost advantage
        for (List<BuildingBlock> group : cornerGroups.values()) {
            if (group.size() >= 2) {
                // Multiple walls on the same tile — splash applies
                int adjacentCount = Math.min(group.size(), RaidConstants.MAX_SPLASH_TARGETS);
                for (BuildingBlock wall : group) {
                    int splashCost = RaidConstants.splashSulfurCostPerWall(wall.getTier(), adjacentCount);
                    int individualCost = getBlockSulfurCost(wall);
                    costs.put(wall, Math.min(splashCost, individualCost));
                }
            }
        }

        return costs;
    }

    private static class DistNode implements Comparable<DistNode> {
        final NodeKey id;
        final int cost;
        DistNode(NodeKey id, int cost) { this.id = id; this.cost = cost; }
        @Override public int compareTo(DistNode o) { return Integer.compare(this.cost, o.cost); }
    }

    /**
     * Dijkstra using sulfur costs to find cheapest raid path.
     */
    private Map<NodeKey, Integer> dijkstraRaid(HouseGraph graph, TileNode start,
                                               Map<BuildingBlock, Integer> splashCosts) {
        Map<NodeKey, Integer> dist = new HashMap<>();
        PriorityQueue<DistNode> queue = new PriorityQueue<>();

        dist.put(start.id, 0);
        queue.add(new DistNode(start.id, 0));

        Map<NodeKey, TileNode> nodeMap = new HashMap<>();
        for (TileNode n : graph.getAllNodes()) {
            nodeMap.put(n.id, n);
        }

        Set<NodeKey> visited = new HashSet<>();

        while (!queue.isEmpty()) {
            DistNode current = queue.poll();
            NodeKey curId = current.id;
            int curCost = current.cost;

            if (visited.contains(curId)) continue;
            visited.add(curId);

            TileNode curNode = nodeMap.get(curId);
            if (curNode == null) continue;

            for (TileEdge edge : graph.getEdges(curNode)) {
                int edgeCost;
                if (edge.blocker != null) {
                    // Fix Dijkstra exploit: Do not use splash cost during path traversal!
                    // A rocket destroys group but we are passing through 1 wall, so raw cost determines min path.
                    edgeCost = edge.raidSulfurCost;
                } else {
                    edgeCost = edge.raidSulfurCost;
                }

                int newCost = curCost + edgeCost;
                NodeKey neighborId = edge.to.id;

                if (newCost < dist.getOrDefault(neighborId, Integer.MAX_VALUE)) {
                    dist.put(neighborId, newCost);
                    queue.add(new DistNode(neighborId, newCost));
                }
            }
        }

        return dist;
    }

    private int getBlockSulfurCost(BuildingBlock block) {
        if (block instanceof Wall && block.getType() == BuildingType.DOORWAY) {
            return ((Wall) block).getDoorType().getSulfurCost() +
                   RaidConstants.getWallSulfurCost(block.getTier());
        }
        return RaidConstants.getWallSulfurCost(block.getTier());
    }

    private boolean isWallType(BuildingBlock b) {
        BuildingType t = b.getType();
        return t == BuildingType.WALL || t == BuildingType.DOORWAY || t == BuildingType.WINDOW_FRAME;
    }
}
