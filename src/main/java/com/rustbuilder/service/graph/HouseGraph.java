package com.rustbuilder.service.graph;

import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;
import com.rustbuilder.service.*;
import com.rustbuilder.service.evaluator.*;
import com.rustbuilder.service.physics.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.service.evaluator.RaidConstants;

/**
 * Builds a 3D adjacency graph from placed building blocks.
 * Nodes = tile positions (tileX, tileY, floorZ).
 * Edges = connections between adjacent tiles through walls/doors/stairs.
 */
public class HouseGraph {

    /** A node in the house graph, representing a tile position. */
    public static class TileNode {
        public final double x, y;
        public final int z;
        public final NodeKey id;
        public final String type; // "tile", "outside", "tc", "workbench", "loot_room"

        public TileNode(double x, double y, int z, String type) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.type = type;
            this.id = type.equals("outside") ? new NodeKey(0, 0, 0, "OUTSIDE") : new NodeKey(x, y, z, type);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TileNode)) return false;
            return id.equals(((TileNode) o).id);
        }

        @Override
        public int hashCode() {
            return id.hashCode();
        }

        @Override
        public String toString() {
            return id.toString();
        }
    }

    /** An edge between two tiles, with walk cost and raid cost. */
    public static class TileEdge {
        public final TileNode from, to;
        public final double walkCost;      // steps to walk (∞ if wall blocks)
        public final int raidSulfurCost;   // sulfur to destroy blocking structure
        public final BuildingBlock blocker; // the wall/floor blocking this edge (null if open)

        public TileEdge(TileNode from, TileNode to, double walkCost, int raidSulfurCost, BuildingBlock blocker) {
            this.from = from;
            this.to = to;
            this.walkCost = walkCost;
            this.raidSulfurCost = raidSulfurCost;
            this.blocker = blocker;
        }
    }

    private final Map<NodeKey, TileNode> nodes = new LinkedHashMap<>();
    private final Map<NodeKey, List<TileEdge>> adjacency = new HashMap<>();
    private final TileNode outsideNode = new TileNode(0, 0, 0, "outside");

    public void buildGraph(List<BuildingBlock> blocks) {
        nodes.clear();
        adjacency.clear();

        // Add outside node
        addNode(outsideNode);

        // 1. Identify all foundation/floor tiles and create nodes
        for (BuildingBlock block : blocks) {
            if (isFoundationOrFloor(block)) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "tile");
                addNode(node);
            }
        }

        // 2. Add special deployable nodes on their tile
        for (BuildingBlock block : blocks) {
            if (block.getType() == BuildingType.TC) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "tc");
                addNode(node);
                // Connect TC to its tile with zero cost
                TileNode tile = findTileNode(block.getX(), block.getY(), block.getZ());
                if (tile != null) {
                    addEdge(node, tile, 0, 0, null);
                    addEdge(tile, node, 0, 0, null);
                }
            } else if (block.getType() == BuildingType.WORKBENCH) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "workbench");
                addNode(node);
                TileNode tile = findTileNode(block.getX(), block.getY(), block.getZ());
                if (tile != null) {
                    addEdge(node, tile, 0, 0, null);
                    addEdge(tile, node, 0, 0, null);
                }
            } else if (block.getType() == BuildingType.LOOT_ROOM) {
                TileNode node = new TileNode(block.getX(), block.getY(), block.getZ(), "loot_room");
                addNode(node);
                TileNode tile = findTileNode(block.getX(), block.getY(), block.getZ());
                if (tile != null) {
                    addEdge(node, tile, 0, 0, null);
                    addEdge(tile, node, 0, 0, null);
                }
            }
        }


        // 3. Build horizontal edges between adjacent tiles (same floor)
        List<TileNode> tileNodes = new ArrayList<>();
        for (TileNode n : nodes.values()) {
            if ("tile".equals(n.type)) tileNodes.add(n);
        }

        for (int i = 0; i < tileNodes.size(); i++) {
            for (int j = i + 1; j < tileNodes.size(); j++) {
                TileNode a = tileNodes.get(i);
                TileNode b = tileNodes.get(j);
                if (a.z != b.z) continue;

                double dist = Math.hypot(a.x - b.x, a.y - b.y);
                if (dist < GameConstants.TILE_SIZE * 1.1) {
                    // Adjacent tiles — check for walls between them
                    BuildingBlock wall = findWallBetween(blocks, a, b);
                    if (wall == null) {
                        // Open passage
                        addEdge(a, b, 1, 0, null);
                        addEdge(b, a, 1, 0, null);
                    } else if (wall.getType() == BuildingType.DOORWAY) {
                        // Door — walkable, automatically assumes a default door is installed for AI simplicity
                        int doorSulfur = 0;
                        if (wall instanceof Wall) {
                             doorSulfur = ((Wall) wall).getDoorType().getSulfurCost();
                        }
                        
                        addEdge(a, b, 2, doorSulfur, wall);
                        addEdge(b, a, 2, doorSulfur, wall);
                    } else {
                        // Solid wall — not walkable, has sulfur cost
                        int sulfur = RaidConstants.getWallSulfurCost(wall.getTier());
                        addEdge(a, b, Double.MAX_VALUE, sulfur, wall);
                        addEdge(b, a, Double.MAX_VALUE, sulfur, wall);
                    }
                }
            }
        }

        // 4. Vertical edges (stairs between floors)
        for (BuildingBlock block : blocks) {
            if (block.getType() == BuildingType.STAIRS) {
                TileNode below = findTileNode(block.getX(), block.getY(), block.getZ());
                TileNode above = findTileNode(block.getX(), block.getY(), block.getZ() + 1);
                if (below != null && above != null) {
                    addEdge(below, above, 3, 0, null);
                    addEdge(above, below, 3, 0, null);
                }
            }
        }

        // 5. Connect outside to tiles on ANY floor that have an exposed edge (no wall on boundary)
        // This simulates using ladders/twigs to reach upper floors
        for (TileNode tile : tileNodes) {
            if (hasExposedEdge(blocks, tile)) {
                addEdge(outsideNode, tile, 1, 0, null);
                addEdge(tile, outsideNode, 1, 0, null);
            }
        }

        // 6. Also: outside connects through outer walls for raiding (on any floor)
        for (TileNode tile : tileNodes) {
             List<BuildingBlock> outerWalls = findOuterWalls(blocks, tile);
            for (BuildingBlock wall : outerWalls) {
                int sulfur;
                double walkCost;
                if (wall.getType() == BuildingType.DOORWAY) {
                    // Automatically assume doorway has a door for AI evaluation
                    if (wall instanceof Wall) {
                        sulfur = ((Wall) wall).getDoorType().getSulfurCost();
                    } else {
                        sulfur = 0;
                    }
                    walkCost = 2; // Can walk through
                } else {
                    sulfur = RaidConstants.getWallSulfurCost(wall.getTier());
                    walkCost = Double.MAX_VALUE; // Can't walk through wall
                }
                // Add edge: check we don't have a cheaper open edge already
                addEdge(outsideNode, tile, walkCost, sulfur, wall);
                addEdge(tile, outsideNode, walkCost, sulfur, wall);
            }
        }

        // 7. Vertical raid edges — through floors/ceilings
        for (BuildingBlock block : blocks) {
            if (isFloor(block) && block.getZ() > 0) {
                TileNode below = findTileNode(block.getX(), block.getY(), block.getZ() - 1);
                TileNode above = findTileNode(block.getX(), block.getY(), block.getZ());
                if (below != null && above != null) {
                    int sulfur = RaidConstants.getWallSulfurCost(block.getTier());
                    // Only add if no stairs already provide free access
                    addEdge(below, above, Double.MAX_VALUE, sulfur, block);
                    addEdge(above, below, Double.MAX_VALUE, sulfur, block);
                }
            }
        }
    }
    
    // Removed dead code findDoorForWall

    // === Graph Query Methods ===

    public TileNode getOutsideNode() {
        return outsideNode;
    }

    public Collection<TileNode> getAllNodes() {
        return nodes.values();
    }

    public List<TileEdge> getEdges(TileNode node) {
        return adjacency.getOrDefault(node.id, Collections.emptyList());
    }

    public TileNode findNodeByType(String type) {
        for (TileNode n : nodes.values()) {
            if (n.type.equals(type)) return n;
        }
        return null;
    }

    public List<TileNode> findAllNodesByType(String type) {
        List<TileNode> result = new ArrayList<>();
        for (TileNode n : nodes.values()) {
            if (n.type.equals(type)) result.add(n);
        }
        return result;
    }

    /**
     * Returns all edges in the graph (for splash damage analysis).
     */
    public List<TileEdge> getAllEdges() {
        List<TileEdge> all = new ArrayList<>();
        for (List<TileEdge> edges : adjacency.values()) {
            all.addAll(edges);
        }
        return all;
    }

    // === Internal Helpers ===

    private void addNode(TileNode node) {
        if (!nodes.containsKey(node.id)) {
            nodes.put(node.id, node);
            adjacency.put(node.id, new ArrayList<>());
        }
    }

    private void addEdge(TileNode from, TileNode to, double walkCost, int raidSulfur, BuildingBlock blocker) {
        adjacency.computeIfAbsent(from.id, k -> new ArrayList<>())
                  .add(new TileEdge(from, to, walkCost, raidSulfur, blocker));
    }

    private TileNode findTileNode(double x, double y, int z) {
        NodeKey id = new NodeKey(x, y, z, "tile");
        return nodes.get(id);
    }

    private boolean isFoundationOrFloor(BuildingBlock b) {
        return BuildingTypeUtils.isHorizontalSurface(b.getType());
    }

    private boolean isFloor(BuildingBlock b) {
        return BuildingTypeUtils.isFloor(b.getType());
    }

    private boolean isWallType(BuildingBlock b) {
        return BuildingTypeUtils.isWall(b.getType());
    }

    /**
     * Find wall between two adjacent tiles by checking if any wall's collision polygon
     * intersects the line segment connecting the centers of Tile A and Tile B.
     */
    private BuildingBlock findWallBetween(List<BuildingBlock> blocks, TileNode a, TileNode b) {
        // Tile centers
        double ax = a.x + GameConstants.HALF_TILE;
        double ay = a.y + GameConstants.HALF_TILE;
        double bx = b.x + GameConstants.HALF_TILE;
        double by = b.y + GameConstants.HALF_TILE;

        for (BuildingBlock block : blocks) {
            if (!isWallType(block)) continue;
            if (block.getZ() != a.z) continue;

            double[] poly = block.getCollisionPoints();
            if (poly == null || poly.length < 8) continue;

            // Check if the line segment A->B intersects any of the 4 edges of the wall's polygon
            for (int i = 0; i < 4; i++) {
                double x1 = poly[i * 2];
                double y1 = poly[i * 2 + 1];
                double x2 = poly[((i + 1) % 4) * 2];
                double y2 = poly[((i + 1) % 4) * 2 + 1];

                if (lineIntersectsLine(ax, ay, bx, by, x1, y1, x2, y2)) {
                    return block;
                }
            }
        }
        return null;
    }

    /**
     * Standard line segment intersection algorithm.
     */
    private boolean lineIntersectsLine(double x1, double y1, double x2, double y2,
                                       double x3, double y3, double x4, double y4) {
        double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (Math.abs(d) < 1e-6) return false;

        double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d;
        double u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d;

        return t >= -0.05 && t <= 1.05 && u >= -0.05 && u <= 1.05;
    }


    private BuildingBlock getBlockForTile(List<BuildingBlock> blocks, TileNode tile) {
        for (BuildingBlock b : blocks) {
            if (isFoundationOrFloor(b) && b.getZ() == tile.z && 
                Math.abs(b.getX() - tile.x) < 0.1 && Math.abs(b.getY() - tile.y) < 0.1) {
                return b;
            }
        }
        return null;
    }

    private List<double[]> getOuterCheckPoints(BuildingBlock tileBlock) {
        List<double[]> checkPoints = new ArrayList<>();
        double[] poly = tileBlock.getCollisionPoints();
        if (poly == null || poly.length < 6) return checkPoints;

        int n = poly.length / 2;
        double cx = tileBlock.getX() + GameConstants.HALF_TILE;
        double cy = tileBlock.getY() + GameConstants.HALF_TILE;
        
        for (int i = 0; i < n; i++) {
            double x1 = poly[i * 2];
            double y1 = poly[i * 2 + 1];
            double x2 = poly[((i + 1) % n) * 2];
            double y2 = poly[((i + 1) % n) * 2 + 1];
            
            double mx = (x1 + x2) / 2.0;
            double my = (y1 + y2) / 2.0;
            
            double dx = mx - cx;
            double dy = my - cy;
            double len = Math.hypot(dx, dy);
            if (len > 0) { dx /= len; dy /= len; }
            
            checkPoints.add(new double[] { cx + dx * 60.0, cy + dy * 60.0 });
        }
        return checkPoints;
    }

    private boolean hasExposedEdge(List<BuildingBlock> blocks, TileNode tile) {
        BuildingBlock tileBlock = getBlockForTile(blocks, tile);
        if (tileBlock == null) return false;
        
        List<double[]> checkPoints = getOuterCheckPoints(tileBlock);
        
        for (double[] pt : checkPoints) {
            boolean hasNeighbor = false;
            for (TileNode other : nodes.values()) {
                if ("tile".equals(other.type) && other.z == tile.z && Math.hypot((other.x+GameConstants.HALF_TILE) - pt[0], (other.y+GameConstants.HALF_TILE) - pt[1]) < 20.0) {
                    hasNeighbor = true;
                    break;
                }
            }
            
            if (!hasNeighbor) {
                TileNode tempNeighbor = new TileNode(pt[0] - GameConstants.HALF_TILE, pt[1] - GameConstants.HALF_TILE, tile.z, "tile");
                BuildingBlock wall = findWallBetween(blocks, tile, tempNeighbor);
                if (wall == null) return true;
            }
        }
        return false;
    }

    private List<BuildingBlock> findOuterWalls(List<BuildingBlock> blocks, TileNode tile) {
        List<BuildingBlock> outerWalls = new ArrayList<>();
        BuildingBlock tileBlock = getBlockForTile(blocks, tile);
        if (tileBlock == null) return outerWalls;
        
        List<double[]> checkPoints = getOuterCheckPoints(tileBlock);
        
        for (double[] pt : checkPoints) {
            boolean hasNeighbor = false;
            for (TileNode other : nodes.values()) {
                if ("tile".equals(other.type) && other.z == tile.z && Math.hypot((other.x+GameConstants.HALF_TILE) - pt[0], (other.y+GameConstants.HALF_TILE) - pt[1]) < 20.0) {
                    hasNeighbor = true;
                    break;
                }
            }
            
            if (!hasNeighbor) {
                TileNode tempNeighbor = new TileNode(pt[0] - GameConstants.HALF_TILE, pt[1] - GameConstants.HALF_TILE, tile.z, "tile");
                BuildingBlock wall = findWallBetween(blocks, tile, tempNeighbor);
                if (wall != null) outerWalls.add(wall);
            }
        }
        return outerWalls;
    }
}
