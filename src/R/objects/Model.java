/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package R.objects;

import R.Mat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

/**
 *
 * @author thanos
 */
public class Model {
    
    private final HashMap<String, Mat> registry;

    public Model() {        
        registry = new HashMap<>(100);
    }
    
    
    public void put(String key, Mat m) {
        registry.put(key, m);
    }
    
    public Mat get(String key) {
        return registry.get(key);
    }
            
    public List<String> keys() {
       
        Set<String> kSet = this.registry.keySet();
        List<String> kList = new ArrayList<>(kSet);
        Collections.sort(kList);
        return kList;
    }
}
