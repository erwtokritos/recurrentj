/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package R;

import R.objects.State;
import R.objects.Model;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author thanos
 */
public class Utils {

    public static Model initLSTM(int input_size, int[] hidden_sizes, int output_size) {

        Model model = new Model();

        //hidden size should be a list        
        for (int d = 0; d < hidden_sizes.length; d++) { // loop over depths

            int prev_size = (d == 0) ? input_size : hidden_sizes[d - 1];
            int hidden_size = hidden_sizes[d];

            //gates parameters
            model.put("Wix" + d, RandMat(hidden_size, prev_size, 0, 0.08));
            model.put("Wih" + d, RandMat(hidden_size, hidden_size, 0, 0.08));
            model.put("bi" + d, new Mat(hidden_size, 1));
            model.put("Wfx" + d, RandMat(hidden_size, prev_size, 0, 0.08));
            model.put("Wfh" + d, RandMat(hidden_size, hidden_size, 0, 0.08));
            //try setting high forger bias
            model.put("bf" + d, Utils.ones(hidden_size, 1));
            //model.put("bf" + d, new Mat(hidden_size, 1));
            model.put("Wox" + d, RandMat(hidden_size, prev_size, 0, 0.08));
            model.put("Woh" + d, RandMat(hidden_size, hidden_size, 0, 0.08));
            model.put("bo" + d, new Mat(hidden_size, 1));

            //cell write params
            model.put("Wcx" + d, RandMat(hidden_size, prev_size, 0, 0.08));
            model.put("Wch" + d, RandMat(hidden_size, hidden_size, 0, 0.08));
            model.put("bc" + d, new Mat(hidden_size, 1));

        }

        //decoder params
        model.put("Whd", RandMat(output_size, hidden_sizes[hidden_sizes.length - 1], 0, 0.08));
        model.put("Wbd", new Mat(output_size, 1));

        return model;
    }

    public static Model initRNN(int input_size, int[] hidden_sizes, int output_size) {

        Model model = new Model();

        //hidden size should be a list        
        for (int d = 0; d < hidden_sizes.length; d++) { // loop over depths

            int prev_size = (d == 0) ? input_size : hidden_sizes[d - 1];
            int hidden_size = hidden_sizes[d];

            model.put("Wxh" + d, RandMat(hidden_size, prev_size, 0, 0.08));
            model.put("Whh" + d, RandMat(hidden_size, hidden_size, 0, 0.08));
            model.put("bhh" + d, new Mat(hidden_size, 1));

        }

        //decoder params
        model.put("Whd", RandMat(output_size, hidden_sizes[hidden_sizes.length - 1], 0, 0.08));
        model.put("Wbd", new Mat(output_size, 1));

        return model;
    }

    public static State forwardLSTM(Graph G, Model model, int[] hidden_sizes, Mat x, State prev) throws Exception {

        //forward prop for a single tick of LSTM
        //G is graph to append ops to
        // model contains LSTM parameters
        // x is 1D columnn vector with observation
        // prev is a struct containing hidden and cell from previous iteration
        List<Mat> hidden_prevs, cell_prevs;

        if (prev == null || prev.h == null) {

            hidden_prevs = new ArrayList<>();
            cell_prevs = new ArrayList<>();
            for (int d = 0; d < hidden_sizes.length; d++) {
                hidden_prevs.add(new Mat(hidden_sizes[d], 1));
                cell_prevs.add(new Mat(hidden_sizes[d], 1));
            }
        } else {

            hidden_prevs = prev.h;
            cell_prevs = prev.c;
        }

        List<Mat> hidden = new ArrayList<>();
        List<Mat> cell = new ArrayList<>();

        for (int d = 0; d < hidden_sizes.length; d++) {

            Mat input_vector = (d == 0) ? x : hidden.get(d - 1);
            Mat hidden_prev = hidden_prevs.get(d);
            Mat cell_prev = cell_prevs.get(d);

            //input gate
            Mat h0 = G.mul(model.get("Wix" + d), input_vector);
            Mat h1 = G.mul(model.get("Wih" + d), hidden_prev);
            Mat input_gate = G.sigmoid(G.add(G.add(h0, h1), model.get("bi" + d)));

            //forget gate
            Mat h2 = G.mul(model.get("Wfx" + d), input_vector);
            Mat h3 = G.mul(model.get("Wfh" + d), hidden_prev);
            Mat forget_gate = G.sigmoid(G.add(G.add(h2, h3), model.get("bf" + d)));

            //output gate
            Mat h4 = G.mul(model.get("Wox" + d), input_vector);
            Mat h5 = G.mul(model.get("Woh" + d), hidden_prev);
            Mat output_gate = G.sigmoid(G.add(G.add(h4, h5), model.get("bo" + d)));

            //write operation on cells
            Mat h6 = G.mul(model.get("Wcx" + d), input_vector);
            Mat h7 = G.mul(model.get("Wch" + d), hidden_prev);
            Mat cell_write = G.tanh(G.add(G.add(h6, h7), model.get("bc" + d)));

            //compute new cell activation
            Mat retain_cell = G.eltmul(forget_gate, cell_prev); //what do we keep from cell
            Mat write_cell = G.eltmul(input_gate, cell_write); //what do we write to cell
            Mat cell_d = G.add(retain_cell, write_cell);  // new cell contents

            //compute hidden state as gated, saturated cell activations
            Mat hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

            hidden.add(hidden_d);
            cell.add(cell_d);

        }

        //one decoder to outputs at end
        Mat output = G.add(G.mul(model.get("Whd"), hidden.get(hidden.size() - 1)), model.get("Wbd"));

        State state = new State();
        state.h = hidden;
        state.c = cell;
        state.o = output;

        return state;

    }

    public static State forwardRNN(Graph G, Model model, int[] hidden_sizes, Mat x, State prev) throws Exception {

        // forward prop for a single tick of RNN
        // G is graph to append ops to
        // model contains LSTM parameters
        // x is 1D columnn vector with observation
        // prev is a struct containing hidden and cell from previous iteration
        List<Mat> hidden_prevs;

        if (prev == null || prev.h == null) {

            hidden_prevs = new ArrayList<>();
            for (int d = 0; d < hidden_sizes.length; d++) {
                hidden_prevs.add(new Mat(hidden_sizes[d], 1));
            }
        } else {

            hidden_prevs = prev.h;
        }

        List<Mat> hidden = new ArrayList<>();

        for (int d = 0; d < hidden_sizes.length; d++) {

            Mat input_vector = (d == 0) ? x : hidden.get(d - 1);
            Mat hidden_prev = hidden_prevs.get(d);

            Mat h0 = G.mul(model.get("Wxh" + d), input_vector);
            Mat h1 = G.mul(model.get("Whh" + d), hidden_prev);
            Mat hidden_d = G.relu(G.add(G.add(h0, h1), model.get("bhh" + d)));

            hidden.add(hidden_d);
        }

        //one decoder to outputs at end
        Mat output = G.add(G.mul(model.get("Whd"), hidden.get(hidden.size() - 1)), model.get("Wbd"));

        State state = new State();
        state.h = hidden;
        state.o = output;

        return state;

    }

    public static Mat RandMat(int n, int d, double mu, double std) {
        Mat m = new Mat(n, d);
        for (int i = 0; i < m.getLength(); i++) {
            m.set(i, randf(-std, std));
        }
        return m;
    }

    /**
     * Returns a random double value between 'low' and 'high'
     *
     * @param low
     * @param high
     * @return
     */
    public static double randf(double low, double high) {
        return Math.random() * (high - low) + low;
    }

    public static Mat RandMat(int n, int d) {
        return RandMat(n, d, 0.0, 0.08);
    }

    public static Mat softmax(final Mat m) {

        Mat out = new Mat(m.n, m.d);
        double maxval = -999999;

        for (int i = 0; i < m.w.length; i++) {
            if (m.w[i] > maxval) {
                maxval = m.w[i];
            }
        }

        double s = 0.0;

        for (int i = 0; i < m.w.length; i++) {
            out.w[i] = Math.exp(m.w[i] - maxval);
            s += out.w[i];
        }

        for (int i = 0; i < m.w.length; i++) {
            out.w[i] /= s;
        }

        return out;
    }
    
    
    public static double log2(double x) {
        return Math.log10(x) / Math.log10(2);
    }
    
    public static double median(List<Double> list) {
        
        List<Double> l = new ArrayList<>();
        l.addAll(list);
        Collections.sort(l);
        if(l.size() % 2 == 0) {
            return l.get(l.size()/2);            
        }else {
            return (l.get((l.size() - 1)/2) + l.get((l.size() + 1)/2))/2;
        }        
    }
    
    public static List<Integer> getSequence(int start, int end) {
       
        List<Integer> list = new ArrayList<>(end - start + 1);
        for(int q = start; q <= end; q++) {
            list.add(q);
        }
        
        return list;
    }

    public static int samplei(double[] w) {
        double r = randf(0, 1);
        double x = 0.0;
        int i = 0;
        while(true) {
            
            x += w[i];
            if(x > r) {
                return i;
            }
            i++;
        }
        /*
        double r = randf(0, 1);
        double x = 0.0;
        List<Integer> seq = getSequence(0, w.length - 1);
        Collections.shuffle(seq);
        for(int i = 0; i < seq.size(); i++) {
            
            x += w[seq.get(i)];
            if(x > r) {               
                return i;
            }            
        }
        
       return w.length - 1;
                */
    }
    
    public static int maxi(double[] w) {
        
        double maxv = w[0];
        int maxix = 0;
        

        for(int i = 1; i < w.length; i++) {
            double v = w[i];
            if(v > maxv) {
                maxv = v;
                maxix = i;
            }
        }
        
        return maxix;
    }    
    
    
   
    public static Mat ones(int n, int d) {
        Mat mat = new Mat(n, d);
        for(int i = 0; i < mat.w.length; i++) {
            mat.w[i] = 1.0;
        }
        
        return mat;
    }
}
