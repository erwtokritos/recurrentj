/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package R;

/*
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
*/
import java.io.IOException;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Mat {
    
    public int n;
    public int d;
    public double[] w;
    public double[] dw;
    Random r = new Random();   

    public Mat(int n, int d) {
        this.n = n;
        this.d = d;
        this.w = new double[n * d];
        this.dw = new double[n * d];        
        
    }
    
    public double get(int row, int col) {
        int ix = coordsToIndex(row , col);
        if((ix < 0) || (ix >= this.w.length )) {
            //throw new Exception("argument out of bounds");
            System.err.println("Argument out of bounds..");
            return 0;
        }
        
        return this.w[ix];        
    }
    
    public double[] getRow(int row) {
        
        double[] data = new double[this.d];
        for(int i = 0; i < this.d; i++ ) {
            data[i] = this.w[row * d + i];            
        }
        return data;
    }
    
    public void set(int row, int col, double value) {
        int ix = coordsToIndex(row , col);
        this.w[ix] = value;        
    }
    
    public void set(int ix, double value) {        
        this.w[ix] = value;
    }
    
    public Mat clone() {
        Mat m = new Mat(this.n, this.d);
        for(int i = 0; i < m.w.length; i++) {
            m.w[i] = this.w[i];
        }
        return m;
    }
    
    public int getLength() {
        return this.w.length;
    }
    
	/*
    public String toJSON() {

        com.google.gson.Gson gson = new Gson();
        String json = gson.toJson(this);

        return json;
    }
     */
    
    
    public void print() {
        for(int r = 0; r < this.n; r++) {
            for(int c = 0; c < this.d; c++) {
                System.out.print(get(r, c) + " ");
            }
            System.out.println();
        }
    }
    
    private int coordsToIndex(int row, int col) {
        return (this.d * row + col);
    }
    
   
}
