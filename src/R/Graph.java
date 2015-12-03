/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package R;

import R.Mat;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import R.objects.Model;

/**
 *
 * @author thanos
 * For the implementation of the backprop step (using a List of Runnable objects), I was inspired by Thomas Lahore's RecurrentJava library 
 * which can be found here 'https://github.com/evolvingstuff/RecurrentJava'
 */
public class Graph {

    boolean needs_backprop;
    List<Runnable> backprop;

    public Graph() {
        this.needs_backprop = true;
        backprop = new ArrayList<>();
    }

    public Graph(boolean needs_backprop) {
        this.needs_backprop = needs_backprop;
        backprop = new ArrayList<>();
    }

    public void backward() {
        for (int i = backprop.size() - 1; i >= 0; i--) {
            backprop.get(i).run();
        }
    }

    public Mat rowPluck(final Mat m, int ix) throws Exception {

        if (ix < 0 || ix >= m.n) {
            throw new Exception("Invalid value for index 'ix'");
        }

        int d = m.d;

        final Mat out = new Mat(d, 1);
        for (int i = 0; i < d; i++) {
            out.w[i] = m.w[d * ix + i];
        }

        if (this.needs_backprop) {
            Runnable bp = new Runnable() {
                public void run() {

                    for (int i = 0; i < d; i++) {
                        m.dw[d * ix + i] += out.dw[i];
                    }

                }
            };
            backprop.add(bp);
        }
        return out;
    }

    public Mat tanh(final Mat m) {

        Mat out = new Mat(m.n, m.d);
        int n = m.w.length;
        for (int i = 0; i < n; i++) {
            out.w[i] = Math.tanh(m.w[i]);
        }

        if (this.needs_backprop) {
            Runnable bp = new Runnable() {

                @Override
                public void run() {
                    for (int i = 0; i < n; i++) {
                        double mwi = out.w[i];
                        m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
                    }
                }
            };

            this.backprop.add(bp);
        }

        return out;
    }

    private double sig(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    public Mat sigmoid(final Mat m) {

        Mat out = new Mat(m.n, m.d);
        int n = m.w.length;
        for (int i = 0; i < n; i++) {
            out.w[i] = sig(m.w[i]);
        }

        if (this.needs_backprop) {
            Runnable bp = new Runnable() {

                @Override
                public void run() {
                    for (int i = 0; i < n; i++) {
                        double mwi = out.w[i];
                        m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
                    }
                }
            };

            this.backprop.add(bp);
        }

        return out;
    }

    public Mat relu(final Mat m) {

        Mat out = new Mat(m.n, m.d);
        int n = m.w.length;
        for (int i = 0; i < n; i++) {
            out.w[i] = Math.max(0.0, m.w[i]);  //relu
        }

        if (this.needs_backprop) {
            Runnable bp = new Runnable() {

                @Override
                public void run() {
                    for (int i = 0; i < n; i++) {

                        m.dw[i] += (m.w[i] > 0) ? out.dw[i] : 0.0;
                    }
                }
            };

            this.backprop.add(bp);
        }

        return out;
    }

    public Mat mul(final Mat m1, final Mat m2) throws Exception {
        if (m1.d != m2.n) {
            String msg = "[" + m1.n + "x" + m1.d + "]  " + "[" + m2.n + "x" + m2.d +"]";
            throw new Exception("matmul dimensions misalligned\n " + msg);
        }

        int n = m1.n;
        int d = m2.d;
        Mat out = new Mat(n, d);

        for (int i = 0; i < m1.n; i++) { // loop over rows of m1            
            for (int j = 0; j < m2.d; j++) { //loop over cols of m2

                double dot = 0.0;
                for (int k = 0; k < m1.d; k++) {  //dot product loop
                    dot += m1.w[m1.d * i + k] * m2.w[m2.d * k + j];

                }
                out.w[d * i + j] = dot;
            }
        }

        if (this.needs_backprop) {
            Runnable bp = new Runnable() {

                @Override
                public void run() {
                    for (int i = 0; i < m1.n; i++) { // loop over rows of m1            
                        for (int j = 0; j < m2.d; j++) { //loop over cols of m2
                            for (int k = 0; k < m1.d; k++) {  //dot product loop
                                double b = out.dw[d * i + j];
                                m1.dw[m1.d * i + k] += m2.w[m2.d * k + j] *b;
                                m2.dw[m2.d * k + j] += m1.w[m1.d * i + k] *b;
                            }                          
                        }
                    }

                }
            };

            this.backprop.add(bp);
        }

        return out;
    }

    public Mat add(final Mat m1, final Mat m2) throws Exception {
        
        if(m1.w.length != m2.w.length) {
            throw new Exception("matadd size mismatch");            
        }
        
        Mat out = new Mat(m1.n, m1.d);
        for(int i = 0; i < m1.w.length; i++) {
            out.w[i] = m1.w[i] + m2.w[i];            
        }
        
        if(this.needs_backprop) {
            Runnable bp = new Runnable() {

                @Override
                public void run() {
                    for(int i = 0; i < m1.w.length; i++ ) {
                        m1.dw[i] += out.dw[i];
                        m2.dw[i] += out.dw[i];
                    }                    
                }
            };
              
            this.backprop.add(bp);                    
        }
        
        return out;
    }
    
    
    public Mat eltmul(final Mat m1, final Mat m2) throws Exception {
       
        if(m1.w.length != m2.w.length) {
            throw new Exception("mateltmul size mismatch");            
        }
        
        final Mat out = new Mat(m1.n, m1.d);
        for (int i = 0; i < m1.w.length; i++) {
            out.w[i] = m1.w[i] * m2.w[i];
        }
        if (this.needs_backprop) {
            Runnable bp = new Runnable() {
                public void run() {
                    for (int i = 0; i < m1.w.length; i++) {
                        m1.dw[i] += m2.w[i] * out.dw[i];
                        m2.dw[i] += m1.w[i] * out.dw[i];
                    }
                }
            };
            backprop.add(bp);
        }
        return out;
    }
    
    

}
