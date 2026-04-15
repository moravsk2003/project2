package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.rl.*;
import com.rustbuilder.ai.rl.legacy.*;
import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;

/**
 * [RL REDESIGN]
 * Policy interface for making decisions in a multi-discrete 5-phase action context.
 */
public interface MultiDiscretePhasePolicy {
    
    /**
     * Executes the orchestration of all 5 decision phases 
     * to produce a single MultiDiscreteAction.
     * Optionally records phase progression if observer is provided.
     */
    MultiDiscreteAction chooseAction(MultiDiscretePhaseContext context, MultiDiscreteStateObserver observer);
}
