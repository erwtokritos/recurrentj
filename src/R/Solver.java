/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package R;

import R.objects.Model;
import R.objects.SolverStats;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author thanos
 */
public class Solver {
    
    double decay_rate = 0.999;
    double smooth_eps = 1e-8;    
    HashMap<String, Mat> step_cache;
    
    public Solver() {
        
        this.step_cache = new HashMap<>();
    }
    
    public SolverStats step(Model model, double step_size, double regc, double clipval) {
     
        //perform parameter update
        int num_clipped = 0;
        int num_tot = 0;
        
        for(String key : model.keys()) {
            
            Mat m = model.get(key); //mat ref
            if(!this.step_cache.containsKey(key)) {
                this.step_cache.put(key, new Mat(m.n, m.d));
            } 
            
            Mat s = this.step_cache.get(key);
            for(int i = 0; i < m.w.length; i++) {
                
                //rmsprop adaptive leaning rate
                double mdwi = m.dw[i];
                s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate ) * mdwi * mdwi;
                
                
                //gradient clip
                if(mdwi > clipval) {                    
                    mdwi = clipval;
                    num_clipped++;
                }
                
                if(mdwi < -clipval) {                    
                    mdwi = -clipval;
                    num_clipped++;
                }                                
                num_tot++;
                
                //update (and regularize)
                m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
                m.dw[i] = 0;  //reset gradients for next iteration                                                                                
            }
        }
        
        
        SolverStats stats = new SolverStats();
        stats.ratio_clipped = num_clipped * 1.0 / num_tot;
        return stats;                
    }
}
