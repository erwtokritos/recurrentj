/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package R.tests;

import R.Graph;
import R.Mat;
import R.Solver;
import R.Utils;
import R.objects.Model;
import R.objects.SolverStats;

/**
 *
 * @author thanos
 */
public class SimpleGraph {
    
    public static void main(String[] args) throws Exception {
        
        
        Mat W = Utils.RandMat(10, 4); // weights Mat
        System.out.println(W.toJSON());
        Mat x = Utils.RandMat(4, 1); // random input Mat
        Mat b = Utils.RandMat(10, 1); // bias vector

// matrix multiply followed by bias offset. h is a Mat
        Graph G = new Graph();        
        Mat h = G.add(G.mul(W, x), b);
// the Graph structure keeps track of the connectivities between Mats

        System.out.println(h.toJSON());
// we can now set the loss on h
        h.dw[0] = 1.0; // say we want the first value to be lower
        System.out.println(h.toJSON());
// propagate all gradients backwards through the graph
// starting with h, all the way down to W,x,b
// i.e. this sets .dw field for W,x,b with the gradients
        G.backward();
        System.out.println(W.toJSON());
        Solver s = new R.Solver();
        Model model = new Model();
        model.put("W", W);
        model.put("b", b);
        
        SolverStats stats = s.step(model, 0.01, 0.0001, 5.0);
        System.out.println(stats.toString());
        System.out.println(W.toJSON());
    }
}
