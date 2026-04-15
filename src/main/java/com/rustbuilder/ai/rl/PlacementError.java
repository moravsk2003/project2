package com.rustbuilder.ai.rl;

/**
 * Enumeration of possible reasons for a placement to fail or be invalid.
 */
public enum PlacementError {
    NONE,
    COLLISION,
    NO_SUPPORT,
    BAD_SOCKET,
    GLOBAL_LIMIT,
    FLOOR_CONSTRAINT,
    UNKNOWN
}
