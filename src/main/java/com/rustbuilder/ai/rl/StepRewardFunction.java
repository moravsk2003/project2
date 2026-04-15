package com.rustbuilder.ai.rl;

import java.util.List;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.service.physics.StabilityService;
import com.rustbuilder.config.GameConstants;

/**
 * Calculates intermediate rewards for each RL step to guide the agent.
 * Rewards are kept small (mostly 0.0-0.5 range) to avoid Q-value explosion.
 */
public class StepRewardFunction {

    public static final double PENALTY_NO_SUPPORT = -0.07;
    public static final double BAD_SOCKET = -0.15;
    public static final double PENALTY_COLLISION = -0.02;
    public static final double PENALTY_FLOOR_CONSTRAINT = -0.04;
    public static final double PENALTY_GENERIC_INVALID = -0.02;

    public static double calculate(GridModel gridModel, BuildAction action, boolean inserted, boolean survived, BuildingBlock placed, PlacementError error) {
        if (!inserted || error != PlacementError.NONE) {
            return switch (error) {
                case NO_SUPPORT -> PENALTY_NO_SUPPORT;
                case BAD_SOCKET -> BAD_SOCKET;
                case COLLISION -> PENALTY_COLLISION;
                case FLOOR_CONSTRAINT -> PENALTY_FLOOR_CONSTRAINT;
                default -> PENALTY_GENERIC_INVALID;
            };
        }
        if (!survived || placed == null) {
            return PENALTY_NO_SUPPORT;
        }

        double reward = 0.04; // Base reward for valid placement

        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        if (blocks.isEmpty()) return reward;

        // ===== 1. Socket Connection Reward =====
        // How many other blocks does this block connect to via sockets?
        int socketConnections = countSocketConnections(placed, blocks);
        boolean isFurniture = placed.getType() == BuildingType.TC || 
                              placed.getType() == BuildingType.WORKBENCH || 
                              placed.getType() == BuildingType.LOOT_ROOM;
        
        if (socketConnections > 0) {
            // +0.30 per connection, capped at 1.2 (4 connections max reward)
            reward += Math.min(socketConnections * 0.10, 0.9);
        } else if (blocks.size() > 1 && !isFurniture) {
            // Structural penalty for starting a new disconnected segment (no socket connection)
            reward -= 0.1;
        }

        // ===== 2. Structural Stability Reward =====
        // Recalculate stability and reward based on block's stability score
        StabilityService.recalculateAll(gridModel);
        double stability = placed.getStability();
        if (stability > 0) {
            // stability ranges 0.0-1.0, reward proportionally
            reward += stability * 0.05; // max +0.1 for full stability
        } else if (!isFoundation(placed)) {
            // Non-foundation block with zero stability = floating, penalize
            reward -= 0.1;
        }

        // ===== 3. Type-specific bonuses (smaller than before) =====
        if (action.actionType == BuildAction.ActionType.TC) {
            reward += 0.4;
        } else if (action.actionType == BuildAction.ActionType.WORKBENCH || 
                   action.actionType == BuildAction.ActionType.LOOT_ROOM) {
            reward += 0.15;
        }

        // ===== 4. Spatial compactness =====
        // Encourage building near existing blocks (compact base)
        if (blocks.size() > 1) {
            double minDist = Double.MAX_VALUE;
            for (BuildingBlock b : blocks) {
                if (b == placed) continue;
                double d = Math.hypot(b.getX() - placed.getX(), b.getY() - placed.getY());
                if (d < minDist) minDist = d;
            }
            double tileSize = GameConstants.TILE_SIZE;
            if (minDist <= tileSize) {
                reward += 0.07; // Adjacent — compact build
            } else if (minDist > tileSize * 3) {
                reward -= 0.1; // Too far — penalize scattered builds
            }
        }

        return reward;
    }

    /**
     * Count how many other blocks this block connects to via socket proximity.
     */
    private static int countSocketConnections(BuildingBlock placed, List<BuildingBlock> allBlocks) {
        int connections = 0;
        List<Socket> placedSockets = placed.getSockets();
        
        for (BuildingBlock other : allBlocks) {
            if (other == placed) continue;
            
            // Quick distance check first
            if (Math.abs(placed.getX() - other.getX()) > GameConstants.TILE_SIZE * 2.5 ||
                Math.abs(placed.getY() - other.getY()) > GameConstants.TILE_SIZE * 2.5) {
                continue;
            }
            
            // Check socket-to-socket proximity
            boolean connected = false;
            for (Socket s1 : placedSockets) {
                for (Socket s2 : other.getSockets()) {
                    double dx = s1.getX() - s2.getX();
                    double dy = s1.getY() - s2.getY();
                    if (dx * dx + dy * dy < 1.3) {
                        connected = true;
                        break;
                    }
                }
                if (connected) break;
            }
            if (connected) connections++;
        }
        return connections;
    }

    private static boolean isFoundation(BuildingBlock b) {
        return b.getType() == BuildingType.FOUNDATION || b.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }
}
