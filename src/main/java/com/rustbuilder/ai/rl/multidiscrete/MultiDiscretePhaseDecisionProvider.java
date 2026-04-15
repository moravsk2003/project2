package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.rl.*;
import com.rustbuilder.ai.rl.legacy.*;
import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;

/**
 * [RL REDESIGN]
 * A strategy contract for providing concrete multi-discrete phase indices 
 * for a specific development context. 
 * 
 * This enables decoupling between WHO makes the decision (heuristic vs learned model)
 * and HOW the policy layer builds the action.
 */
public interface MultiDiscretePhaseDecisionProvider {
    
    /**
     * Supplies chosen phase indices for a given situation.
     */
    MultiDiscretePhaseDecision provideDecision(MultiDiscretePhaseContext context);
}
