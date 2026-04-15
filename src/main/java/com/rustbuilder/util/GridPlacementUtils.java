package com.rustbuilder.util;

import com.rustbuilder.config.GameConstants;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;

import com.rustbuilder.ai.rl.PlacementError;

public class GridPlacementUtils {

    public static class Placement {
        public double x, y, rotation;
        public Orientation orientation;
        public boolean valid;
        public PlacementError error = PlacementError.NONE;
        
        public Placement(double x, double y, double rotation, Orientation orientation, boolean valid, PlacementError error) {
            this.x = x;
            this.y = y;
            this.rotation = rotation;
            this.orientation = orientation;
            this.valid = valid;
            this.error = error;
        }

        public Placement(double x, double y, double rotation, Orientation orientation, boolean valid) {
            this(x, y, rotation, orientation, valid, PlacementError.NONE);
        }
    }

    public static Placement calculatePlacement(GridModel grid, BuildAction action) {
        double tileSize = GameConstants.TILE_SIZE;
        double halfTile = GameConstants.HALF_TILE;
        double startX = 200; 
        double startY = 200;

        // Note: Building models are centered on top-left X/Y actually
        // gridX, gridY means the center of the grid cell
        double rawCenterX = startX + action.gridX * tileSize;
        double rawCenterY = startY + action.gridY * tileSize;
        
        // Base X,Y of a block if placed exactly on grid
        double exactX = rawCenterX - halfTile;
        double exactY = rawCenterY - halfTile;
        
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;

        BuildingBlock target = null;
        double minDist = Double.MAX_VALUE;
        // Find closest block
        for (BuildingBlock b : grid.getAllBlocks()) {
            if (b.getZ() != z && b.getZ() != z - 1) continue; // Only same floor or floor below
            
            // Measure from center of block to rawCenter
            double d = Math.hypot(b.getX() + halfTile - rawCenterX, b.getY() + halfTile - rawCenterY);
            if (d < minDist) { 
                minDist = d; 
                target = b; 
            }
        }

        boolean isFirst = target == null || minDist > tileSize * 1.5;

        // 1. Initial / Free placement
        if (isFirst) {
            if (action.actionType == BuildAction.ActionType.FOUNDATION) {
                return new Placement(exactX, exactY, 0, Orientation.NORTH, true, PlacementError.NONE);
            }
            if (action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) {
                return new Placement(exactX, exactY, 0, Orientation.NORTH, true, PlacementError.NONE);
            }
            return new Placement(0, 0, 0, Orientation.NORTH, false, PlacementError.BAD_SOCKET);
        }

        // 2. Decor / Deployables (TC, Workbench, Loot) -> Center of Target
        if (action.actionType == BuildAction.ActionType.TC || action.actionType == BuildAction.ActionType.WORKBENCH || action.actionType == BuildAction.ActionType.LOOT_ROOM) {
            if (target != null) {
                return new Placement(target.getX(), target.getY(), target.getRotation(), Orientation.NORTH, true, PlacementError.NONE);
            }
            return new Placement(0, 0, 0, Orientation.NORTH, false, PlacementError.BAD_SOCKET);
        }

        // 3. Doors -> On Doorway
        if (action.actionType == BuildAction.ActionType.DOOR) {
            if (target != null && target.getType() == BuildingType.DOORWAY) {
                return new Placement(target.getX(), target.getY(), target.getRotation(), ((Wall)target).getOrientation(), true, PlacementError.NONE);
            }
            return new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET);
        }

        // 4. Walls / Doorways / Windows
        if (isWallLike(action.actionType)) {
            if (target != null && isValidBase(target.getType(), false)) {
                Orientation orient = getOrientationFromAction(target.getType(), action.orientation);
                return new Placement(target.getX(), target.getY(), target.getRotation(), orient, true);
            }
            return new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET); // Invalid attachment target for wall
        }

        // 5. Connecting Foundations/Floors mathematically
        if (target != null && isValidBase(target.getType(), true)) {
            // Instantiate a dummy in the center of the AI's requested cell
            BuildingBlock dummy = instantiateDummy(action.actionType, exactX, exactY, z);
            if (dummy == null) return new Placement(0,0,0,null,false);
            
            // Try 4 orientations to find the one that best connects sockets mathematically
            double bestShiftX = 0;
            double bestShiftY = 0;
            double bestRot = 0;
            double globalMinDist = Double.MAX_VALUE;
            
            for (int rotGuess = 0; rotGuess < 4; rotGuess++) {
                 double testRot = target.getRotation() + rotGuess * 90;
                 if (action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FLOOR) {
                     testRot = target.getRotation() + rotGuess * 60; // Triangles have 6 faces theoretically
                 }
                 dummy.setRotation((testRot + 360) % 360);
                 
                 for (Socket tSock : target.getSockets()) {
                     if (tSock.getSide() == 10) continue; // Skip center socket
                     for (Socket dSock : dummy.getSockets()) {
                         if (dSock.getSide() == 10) continue;
                         double dist = Math.hypot(tSock.getX() - dSock.getX(), tSock.getY() - dSock.getY());
                         if (dist < globalMinDist) {
                             globalMinDist = dist;
                             bestShiftX = tSock.getX() - dSock.getX();
                             bestShiftY = tSock.getY() - dSock.getY();
                             bestRot = dummy.getRotation();
                         }
                     }
                 }
            }
            
            if (globalMinDist < tileSize * 0.3) {
                 double finalSnapX = exactX + bestShiftX;
                 double finalSnapY = exactY + bestShiftY;
                 
                 // Post-snap overlap guard: center-to-center distance must be >= 0.9 * tileSize
                 double cx1 = target.getX() + halfTile;
                 double cy1 = target.getY() + halfTile;
                 double cx2 = finalSnapX + halfTile;
                 double cy2 = finalSnapY + halfTile;
                 double centerDist = Math.hypot(cx2 - cx1, cy2 - cy1);
                 
                 if (centerDist >= tileSize * 0.9) {
                     return new Placement(finalSnapX, finalSnapY, bestRot, Orientation.NORTH, true);
                 }
            }
        }

        return new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET);
    }
    
    private static boolean isWallLike(BuildAction.ActionType type) {
        return type == BuildAction.ActionType.WALL || type == BuildAction.ActionType.DOORWAY || type == BuildAction.ActionType.WINDOW_FRAME;
    }
    
    private static boolean isValidBase(BuildingType type, boolean includeWall) {
        if (type == BuildingType.FOUNDATION || type == BuildingType.TRIANGLE_FOUNDATION || type == BuildingType.FLOOR || type == BuildingType.TRIANGLE_FLOOR) return true;
        if (includeWall && (type == BuildingType.WALL || type == BuildingType.DOORWAY || type == BuildingType.WINDOW_FRAME)) return true;
        return false;
    }

    private static Orientation getOrientationFromAction(BuildingType baseType, int orientationSelection) {
        if (baseType == BuildingType.TRIANGLE_FOUNDATION || baseType == BuildingType.TRIANGLE_FLOOR) {
            Orientation[] triOrients = {Orientation.TRIANGLE_BASE, Orientation.TRIANGLE_LEFT, Orientation.TRIANGLE_RIGHT};
            return triOrients[orientationSelection % 3];
        } else {
            Orientation[] sqOrients = {Orientation.NORTH, Orientation.EAST, Orientation.SOUTH, Orientation.WEST};
            return sqOrients[orientationSelection % 4];
        }
    }
    
    private static BuildingBlock instantiateDummy(BuildAction.ActionType type, double x, double y, int z) {
        switch (type) {
             case FOUNDATION: return new Foundation(x, y, z);
             case TRIANGLE_FOUNDATION: return new TriangleFoundation(x, y, z, 0);
             case FLOOR: return new Floor(x, y, z, 0);
             case TRIANGLE_FLOOR: return new TriangleFloor(x, y, z, 0);
             default: return null;
        }
    }
}
