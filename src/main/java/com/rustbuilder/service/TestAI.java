package com.rustbuilder.service;

import com.rustbuilder.service.evaluator.HouseEvaluator;

import com.rustbuilder.service.generator.GeneratorService;
import com.rustbuilder.model.core.*;

import com.rustbuilder.ai.ea.GeneticAlgorithmService;

public class TestAI {
    public static void main(String[] args) {
        GeneticAlgorithmService ga = new GeneticAlgorithmService();
        ga.setPopulationSize(50);
        
        System.out.println("Running 500 generations...");
        for (int i = 0; i < 500; i++) {
            ga.evolve(1, 0.33, 0.33, 0.33, null);
            if ((i + 1) % 50 == 0) {
                System.out.printf("Gen %d best fitness: %.4f\n", (i+1), ga.getBestFitness());
            }
        }
        
        com.rustbuilder.ai.ea.BaseGenome best = ga.getBestGenome();
        System.out.println("Best fitness: " + ga.getBestFitness());
        
        com.rustbuilder.model.GridModel tempGrid = new com.rustbuilder.model.GridModel();
        best.decode(tempGrid);
        
        java.util.List<com.rustbuilder.model.core.BuildingBlock> blocks = tempGrid.getAllBlocks();
        System.out.println("Total blocks placed: " + blocks.size());
        
        for (com.rustbuilder.model.core.BuildingBlock b : blocks) {
            String orientationInfo = "";
            if (b instanceof com.rustbuilder.model.structure.Wall) {
                orientationInfo = " Orientation: " + ((com.rustbuilder.model.structure.Wall)b).getOrientation();
            }
            System.out.printf("- %s at (%.1f, %.1f) z=%d rot=%.1f%s\n", 
                b.getType(), b.getX(), b.getY(), b.getZ(), b.getRotation(), orientationInfo);
        }
        
        HouseEvaluator eval = new HouseEvaluator();
        HouseEvaluator.EvaluationResult res = eval.evaluate(tempGrid);
        System.out.println("Logistics: " + res.logistics.score);
        System.out.println("Cost: " + res.cost.score);
        System.out.println("Raid: " + res.raid.score);
        System.out.println("Final: " + res.finalScore);
    }
}
